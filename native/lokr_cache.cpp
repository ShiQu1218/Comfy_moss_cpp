// Reuse MossInfer's exact runtime merge; unchanged tensor bytes and the header are copied verbatim.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include "models/lora/lora.h"
#include "util/json.h"
#include "weights/safetensors.h"

namespace fs = std::filesystem;
using mi::weights::DType;

template <class T>
std::vector<T> read(std::fstream& file, std::uint64_t offset, std::size_t count) {
  std::vector<T> values(count);
  file.seekg(static_cast<std::streamoff>(offset));
  file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (!file) throw std::runtime_error("cache tensor read failed");
  return values;
}

template <class T>
void write(std::fstream& file, std::uint64_t offset, const std::vector<T>& values) {
  file.seekp(static_cast<std::streamoff>(offset));
  file.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
  if (!file) throw std::runtime_error("cache tensor write failed");
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  try {
    if (argc != 2) throw std::runtime_error("usage: moss-lokr-cache manifest.json");
    std::ifstream config(fs::path(argv[1]), std::ios::binary);
    if (!config) throw std::runtime_error("manifest not found");
    const std::string text((std::istreambuf_iterator<char>(config)), {});
    const auto manifest = mi::json::parse(text);
    const auto source = mi::util::path_from_utf8(manifest.at("source").as_string());
    const auto output = mi::util::path_from_utf8(manifest.at("output").as_string());
    if (fs::exists(output)) throw std::runtime_error("output must be a new file");
    mi::weights::SafetensorsFile tensors(source);
    std::vector<mi::lora::LoraSpec> specs;
    for (const auto& spec : manifest.at("loras").as_array())
      specs.push_back({mi::util::path_from_utf8(spec.at("path").as_string()), static_cast<float>(spec.at("strength").as_double())});
    mi::lora::LoraStack stack(specs, mi::lora::moss_dit_targets());
    for (const auto& report : stack.reports())
      if (!report.unknown.empty() || report.adapters == 0) throw std::runtime_error("adapter was not fully recognized");
    for (const auto& adapter : stack.adapters())
      if (adapter.kind != mi::lora::AdapterKind::kLoKr && adapter.kind != mi::lora::AdapterKind::kFull)
        throw std::runtime_error("only LoKr or materialized adapters can be baked by this helper");
    fs::copy_file(source, output);
    // copy_file keeps a read-only source's attributes; the copy must be writable to merge in place.
    fs::permissions(output, fs::perms::owner_write, fs::perm_options::add);
    std::fstream file(output, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) throw std::runtime_error("cache output could not be opened");
    for (const auto& module : stack.merge_modules()) {
      const auto* target = stack.model().find(module);
      if (!target) throw std::runtime_error("unknown LoKr target");
      std::string stored = module;
      std::int64_t row = 0;
      const auto* weight = tensors.find(stored + ".weight");
      if (!weight && (module.ends_with(".proj.q") || module.ends_with(".proj.k") || module.ends_with(".proj.v"))) {
        row = (module.back() == 'q' ? 0 : module.back() == 'k' ? 1 : 2) * target->n;
        stored = module.substr(0, module.size() - 1) + "qkv";
        weight = tensors.find(stored + ".weight");
      }
      if (!weight || weight->shape.size() != 2 || weight->shape[1] != target->k || row + target->n > weight->shape[0])
        throw std::runtime_error("incompatible base weight shape: " + module);
      const auto count = static_cast<std::size_t>(target->n * target->k);
      if (weight->dtype == DType::kI8) {
        const auto* scale = tensors.find(stored + ".weight_scale");
        const auto* config_tensor = tensors.find(stored + ".comfy_quant");
        if (!scale || scale->dtype != DType::kF32 || scale->numel() != weight->shape[0] || !config_tensor || config_tensor->nbytes > 4096)
          throw std::runtime_error("unsupported INT8 scale/config: " + module);
        const auto bytes = read<char>(file, config_tensor->offset, static_cast<std::size_t>(config_tensor->nbytes));
        const auto quant = mi::json::parse(std::string(bytes.begin(), bytes.end()));
        if (quant.at("format").as_string() != "int8_tensorwise" || !quant.at("convrot").as_bool())
          throw std::runtime_error("cache requires ConvRot INT8");
        const int group = static_cast<int>(quant.at("convrot_groupsize").as_int());
        const auto q_offset = weight->offset + static_cast<std::uint64_t>(row * target->k);
        const auto s_offset = scale->offset + static_cast<std::uint64_t>(row * sizeof(float));
        auto q = read<std::int8_t>(file, q_offset, count);
        auto s = read<float>(file, s_offset, static_cast<std::size_t>(target->n));
        mi::lora::merge_requantize_int8(stack, module, q.data(), s.data(), target->n, target->k, group);
        write(file, q_offset, q);
        write(file, s_offset, s);
      } else if (weight->dtype == DType::kBF16) {
        const auto offset = weight->offset + static_cast<std::uint64_t>(row * target->k * 2);
        auto w = read<std::uint16_t>(file, offset, count);
        mi::lora::merge_bf16(stack, module, w.data(), target->n, target->k);
        write(file, offset, w);
      } else throw std::runtime_error("cache supports INT8/BF16 weights only");
      std::cerr << "merged " << module << '\n';
    }
    file.flush();
    if (!file) throw std::runtime_error("cache flush failed");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
