"""Synthetic algebra/numerics checks; does not load models or change runtime files."""
import json

import numpy as np


H4 = np.ones((4, 4), dtype=np.float64) - 2 * np.fliplr(np.eye(4))


def rotation(group):
    h = np.ones((1, 1))
    while h.shape[0] < group:
        h = np.kron(h, H4 / 2)
    assert h.shape == (group, group)
    return h


def dense(x, group):
    """Mirror helper's per-output FP32 accumulation order, without FMA."""
    h = rotation(group).astype(x.dtype)
    blocks = x.reshape(-1, group)
    out = np.zeros_like(blocks)
    for j in range(group):
        out += blocks[:, j:j + 1] * h[j]
    return out.reshape(x.shape)


def butterfly(x, group):
    """Regular H4, not the usual Sylvester H2 transform."""
    out = x.copy()
    stride = 1
    while stride < group:
        v = out.reshape(-1, 4, stride)
        a, b, c, d = (v[:, i].copy() for i in range(4))
        s, t = a + b, a - b
        v[:, 0] = ((s + c) - d) * 0.5
        v[:, 1] = ((s - c) + d) * 0.5
        v[:, 2] = ((t + c) + d) * 0.5
        v[:, 3] = ((-t + c) + d) * 0.5
        stride *= 4
    return out


def quantize(x):
    assert np.isfinite(x).all()
    scale = np.maximum(np.max(np.abs(x), axis=1) * np.float32(1 / 127), np.float32(1e-30))
    q = np.clip(np.rint(x / scale[:, None]), -128, 127).astype(np.int8)
    return q, scale


def difference(reference, candidate):
    rq, rs = quantize(reference)
    cq, cs = quantize(candidate)
    return {
        "max_abs_before_quant": float(np.max(np.abs(reference - candidate))),
        "fp32_different": int(np.count_nonzero(reference.view(np.uint32) != candidate.view(np.uint32))),
        "int8_different": int(np.count_nonzero(rq != cq)),
        "scale_bits_different": int(np.count_nonzero(rs.view(np.uint32) != cs.view(np.uint32))),
        "elements": reference.size,
        "rows": reference.shape[0],
    }


def main():
    rng = np.random.default_rng(71)
    results = {"seed": 71, "numpy": np.__version__, "groups": []}
    for group in (4, 16, 64, 256, 1024):
        h = rotation(group)
        # Dyadic entries make this exact: H is symmetric and its own inverse.
        assert np.array_equal(h, h.T)
        assert np.array_equal(h @ h, np.eye(group))
        x64 = rng.standard_normal((4, group))
        np.testing.assert_allclose(butterfly(x64, group), x64 @ h, rtol=1e-12, atol=1e-12)
        z64, delta64 = x64, rng.standard_normal(x64.shape)
        np.testing.assert_allclose((z64 @ h + delta64) @ h, z64 + delta64 @ h, rtol=1e-12, atol=1e-12)
        integers = rng.integers(-32, 33, (4, group)).astype(np.float64)
        assert np.array_equal(butterfly(integers, group), integers @ h)

        q = rng.integers(-127, 128, (32, 2 * group), dtype=np.int16).astype(np.float32)
        scales = rng.uniform(0.0001, 0.01, (32, 1)).astype(np.float32)
        z = q * scales
        delta = rng.normal(0, 0.03, z.shape).astype(np.float32)
        reference = dense(dense(z, group) + delta, group)
        fast = butterfly(butterfly(z, group) + delta, group)
        rotated_delta = z + butterfly(delta, group)
        results["groups"].append({
            "group": group,
            "butterfly": difference(reference, fast),
            "rotated_delta": difference(reference, rotated_delta),
        })

    # Group-aligned Kronecker factor rotation: (A kron B) R = A kron (B R_d).
    group = 16
    a, b = rng.normal(size=(3, 4)), rng.normal(size=(5, 2 * group))
    full = np.kron(a, b)
    np.testing.assert_allclose(butterfly(full, group), np.kron(a, butterfly(b, group)), rtol=1e-12, atol=1e-12)
    # Kronecker matvec without materializing A kron B, using C-order flattening.
    x = rng.normal(size=(a.shape[1], b.shape[1]))
    np.testing.assert_allclose(full @ x.ravel(), (a @ x @ b.T).ravel(), rtol=1e-12, atol=1e-12)

    # A tiny input change can cross a quantization tie while the row maximum stays fixed.
    edge = np.array([[127, 0.5, -0.5]], dtype=np.float32)
    perturbed = edge.copy()
    perturbed[0, 1] = np.nextafter(edge[0, 1], np.float32(np.inf))
    assert quantize(edge)[0][0, 1] == 0 and quantize(perturbed)[0][0, 1] == 1

    # CFG can be combined before converting velocity to denoised prediction.
    x, vc, vn, vs = rng.normal(size=(4, 128))
    sigma, cfg, slg = 0.7, 4.0, 0.2
    pc, pn, ps = (x - sigma * v for v in (vc, vn, vs))
    original = pn + cfg * (pc - pn) + slg * (pc - ps)
    transformed = x - sigma * (vn + cfg * (vc - vn) + slg * (vc - vs))
    np.testing.assert_allclose(original, transformed, rtol=1e-12, atol=1e-12)
    results["algebra_checks_passed"] = True
    results["quantization_tie_counterexample"] = True
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
