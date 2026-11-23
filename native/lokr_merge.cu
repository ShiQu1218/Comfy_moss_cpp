// Reuse upstream validation, factor preparation and CPU fallback without duplicating them.
#define merge_requantize_int8 merge_requantize_int8_cpu
#include "models/lora/lora_merge.cpp"
#undef merge_requantize_int8
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

namespace mi::lora {
namespace {
void checked(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
template<class T> struct Device {
  T* p = nullptr;
  explicit Device(size_t n) { checked(cudaMalloc(&p, n * sizeof(T))); }
  ~Device() { cudaFree(p); }
  Device(const Device&) = delete;
};
struct Workspace {
  Workspace(int capacity, int rows) : capacity(capacity), x(capacity), y(capacity), delta(capacity),
                                      scale(rows), q(capacity), bad(1) {}
  int capacity;
  int group = 0;
  Device<float> x, y, delta, scale;
  Device<signed char> q;
  Device<int> bad;
  std::unique_ptr<Device<float>> rotation;
};
__global__ void decode(const signed char* q, const float* s, float* w, int count, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) w[i] = float(q[i]) * s[i / k];
}
__global__ void rotate(const float* x, float* y, const float* h, int count, int g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int start = i / g * g, col = i % g;
  float acc = 0;
  for (int j = 0; j < g; ++j) acc += x[start + j] * h[j * g + col];
  y[i] = acc;
}
__global__ void rotate256(const float* x, float* y) {
  __shared__ float stage[2][256];
  const int lane = threadIdx.x;
  const int i = blockIdx.x * 256 + lane;
  stage[0][lane] = x[i];
  __syncthreads();
  // Each stage applies H4/2 along one base-4 digit; alternate buffers to avoid read/write races.
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 2 * pass;
    const int digit = (lane >> shift) & 3;
    const int base = lane & ~(3 << shift);
    float sum = 0;
    for (int j = 0; j < 4; ++j) {
      const float v = stage[pass % 2][base + (j << shift)];
      sum += j + digit == 3 ? -v : v;
    }
    const float value = sum * 0.5f;
    if (pass == 3) y[i] = value;
    else {
      stage[(pass + 1) % 2][lane] = value;
      __syncthreads();
    }
  }
}
__global__ void kron_add(float* w, const float* a, const float* b, int count, int k,
                         int row, int b1, int c2, int d2, float coef) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int r = row + i / k, c = i % k;
  float delta = a[(r / c2) * b1 + c / d2] * b[(r % c2) * d2 + c % d2];
  w[i] = w[i] + coef * delta;
}
__global__ void quantize(const float* w, signed char* q, float* s, int k, int* bad) {
  __shared__ float maxima[256];
  int r = blockIdx.x, t = threadIdx.x;
  float a = 0;
  for (int c = t; c < k; c += 256) {
    float v = w[r * k + c];
    if (!isfinite(v)) atomicExch(bad, 1);
    a = fmaxf(a, fabsf(v));
  }
  maxima[t] = a;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d) maxima[t] = fmaxf(maxima[t], maxima[t + d]);
    __syncthreads();
  }
  float scale = fmaxf(maxima[0] * (1.0f / 127.0f), 1e-30f);
  if (!t) s[r] = scale;
  for (int c = t; c < k; c += 256)
    q[r * k + c] = (signed char)fminf(127, fmaxf(-128, nearbyintf(w[r * k + c] / scale)));
}
__global__ void full_add(float* w, const float* delta, int count, float coef) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) w[i] = w[i] + coef * delta[i];
}
}
int merge_requantize_int8(const LoraStack& stack, std::string_view module, std::int8_t* q, float* scale,
                          std::int64_t n, std::int64_t k, int group, int threads = 0) {
  int devices = 0;
  if (std::getenv("MOSS_LOKR_CPU") || cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
    return merge_requantize_int8_cpu(stack, module, q, scale, n, k, group, threads);
  target(stack, module, n, k);
  if (group && (!quant::valid_convrot_group(group) || k % group)) fail("invalid ConvRot group");
  auto ps = prepare(stack, module, false);
  if (ps.empty()) return 0;
  for (const auto& p : ps)
    if ((p.a->kind != AdapterKind::kLoKr && p.a->kind != AdapterKind::kFull) || !p.bias.empty())
      return merge_requantize_int8_cpu(stack, module, q, scale, n, k, group, threads);
  // Commit host output only after the entire module succeeds, so fallback always starts pristine.
  std::vector<std::int8_t> result(n * k);
  std::vector<float> scales(n);
  // The helper processes its modules serially on the same device.
  static thread_local std::unique_ptr<Workspace> workspace;
  const bool fast_rotation = group == 256 && std::getenv("MOSS_LOKR_FAST_ROTATION");
  try {
    const int rows = 512, capacity = rows * int(k);
    if (!workspace || workspace->capacity < capacity) {
      workspace.reset();
      workspace = std::make_unique<Workspace>(capacity, rows);
    }
    auto& [stored_capacity, stored_group, x, y, delta, ds, dq, bad, rotation] = *workspace;
    checked(cudaMemset(bad.p, 0, sizeof(int)));
    if (group && !fast_rotation && stored_group != group) {
      rotation = std::make_unique<Device<float>>(group * group);
      std::vector<float> table(group * group);
      for (int i = 0; i < group; ++i) for (int j = 0; j < group; ++j)
        table[i * group + j] = quant::hadamard_sign(group, i, j) / std::sqrt(float(group));
      checked(cudaMemcpy(rotation->p, table.data(), table.size() * 4, cudaMemcpyHostToDevice));
      stored_group = group;
    }
    std::vector<std::unique_ptr<Device<float>>> factors;
    for (const auto& p : ps) for (const auto* v : {&p.kw1, &p.kw2}) {
      factors.push_back(std::make_unique<Device<float>>(std::max(size_t(1), v->size())));
      if (!v->empty()) checked(cudaMemcpy(factors.back()->p, v->data(), v->size() * 4, cudaMemcpyHostToDevice));
    }
    for (int r = 0; r < n; r += rows) {
      int nr = int(std::min<std::int64_t>(rows, n - r)), count = nr * int(k), blocks = (count + 255) / 256;
      checked(cudaMemcpy(dq.p, q + r * k, count, cudaMemcpyHostToDevice));
      checked(cudaMemcpy(ds.p, scale + r, nr * 4, cudaMemcpyHostToDevice));
      decode<<<blocks, 256>>>(dq.p, ds.p, x.p, count, int(k));
      float* w = x.p;
      if (group) {
        if (fast_rotation) rotate256<<<count / 256, 256>>>(x.p, y.p);
        else rotate<<<blocks, 256>>>(x.p, y.p, rotation->p, count, group);
        w = y.p;
      }
      for (size_t i = 0; i < ps.size(); ++i) {
        const auto& p = ps[i];
        if (p.a->kind == AdapterKind::kFull) {
          if (!p.a->parts.empty()) {
            auto host = stack.read(p.a->parts[0], r + p.a->row0, nr);
            checked(cudaMemcpy(delta.p, host.data(), count * sizeof(float), cudaMemcpyHostToDevice));
            full_add<<<blocks, 256>>>(w, delta.p, count, p.coef);
          }
          continue;
        }
        kron_add<<<blocks, 256>>>(w, factors[2*i]->p, factors[2*i+1]->p, count, int(k),
          int(r + p.a->row0), int(p.b1), int(p.c2), int(p.d2), p.coef);
      }
      if (fast_rotation) rotate256<<<count / 256, 256>>>(y.p, x.p);
      else if (group) rotate<<<blocks, 256>>>(y.p, x.p, rotation->p, count, group);
      quantize<<<nr, 256>>>(x.p, dq.p, ds.p, int(k), bad.p);
      checked(cudaGetLastError());
      checked(cudaMemcpy(result.data() + r * k, dq.p, count, cudaMemcpyDeviceToHost));
      checked(cudaMemcpy(scales.data() + r, ds.p, nr * 4, cudaMemcpyDeviceToHost));
    }
    int invalid;
    checked(cudaMemcpy(&invalid, bad.p, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid) fail("non-finite GPU merged weight");
  } catch (const std::exception& e) {
    workspace.reset();
    std::cerr << "GPU merge fallback: " << e.what() << '\n';
    return merge_requantize_int8_cpu(stack, module, q, scale, n, k, group, threads);
  }
  std::copy(result.begin(), result.end(), q);
  std::copy(scales.begin(), scales.end(), scale);
  std::cerr << "GPU merged " << module;
  if (fast_rotation) std::cerr << " (fast rotation)";
  std::cerr << '\n';
  return int(ps.size());
}
}
