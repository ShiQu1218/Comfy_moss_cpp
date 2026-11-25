"""Read native SLG settings and combine component predictions in ComfyUI space."""
import inspect
from pathlib import Path

from comfy_extras import nodes_slg


def native_guidance(options):
    if set(options) - {"transformer_options", "sampler_calc_cond_batch_function", "sampler_post_cfg_function"} or options.get("transformer_options", {}):
        raise ValueError("目前僅支援原生跳層引導，無法轉送其他 MODEL patch")
    result = {"simple": None, "post": []}
    callbacks = [("simple", options["sampler_calc_cond_batch_function"])] if "sampler_calc_cond_batch_function" in options else []
    callbacks += [("post", callback) for callback in options.get("sampler_post_cfg_function", [])]
    names = {"simple": "SkipLayerGuidanceDiTSimple.execute.<locals>.calc_cond_batch_function",
             "post": "SkipLayerGuidanceDiT.execute.<locals>.post_cfg_function"}
    for kind, callback in callbacks:
        if not inspect.isfunction(callback) or callback.__qualname__ != names[kind] or Path(callback.__code__.co_filename).resolve() != Path(nodes_slg.__file__).resolve():
            raise ValueError("無法辨識此 MODEL callback；僅支援原生 SkipLayerGuidanceDiT／Simple")
        values = inspect.getclosurevars(callback).nonlocals
        required = {"single_layers", "sigma_start", "sigma_end"} | ({"scale", "rescaling_scale"} if kind == "post" else set())
        if not required <= values.keys():
            raise ValueError("原生 SLG 介面已改變，無法安全轉送")
        # MOSS has single_block hooks only; double_block indices do nothing in the native model too.
        spec = {"layers": sorted({int(layer) for layer in values["single_layers"] if 0 <= int(layer) < 42}),
                "sigma_start": float(values["sigma_start"]), "sigma_end": float(values["sigma_end"])}
        if kind == "simple":
            # Without a valid layer the skipped uncond equals the plain uncond.
            if spec["layers"]:
                result["simple"] = spec
        else:
            spec.update(scale=float(values["scale"]), rescaling_scale=float(values["rescaling_scale"]))
            if spec["scale"] > 0:  # native skips the whole callback body otherwise
                result["post"].append(spec)
    return result


def active(spec, sigma):
    return spec is not None and spec["sigma_end"] <= sigma <= spec["sigma_start"]


def post_rows(guidance):
    """The prediction row of each post spec, or None for one without valid layers (its prediction is cond)."""
    row = 3 if guidance["simple"] is not None else 2
    rows = []
    for spec in guidance["post"]:
        rows.append(row if spec["layers"] else None)
        row += bool(spec["layers"])
    return rows


def prediction_rows(guidance, sigma, cfg):
    negative = 2 if active(guidance["simple"], sigma) else 1
    rows = [0] + ([negative] if cfg != 1 else [])
    rows += [row for row, spec in zip(post_rows(guidance), guidance["post"])
             if row is not None and spec["scale"] > 0 and active(spec, sigma)]
    return rows


def combine_predictions(predictions, guidance, sigma, cfg):
    cond = predictions[0]
    negative = 2 if active(guidance["simple"], sigma) else 1
    result = cond if cfg == 1 else predictions[negative] + (cond - predictions[negative]) * cfg
    for row, spec in zip(post_rows(guidance), guidance["post"]):
        if spec["scale"] > 0 and active(spec, sigma):
            if row is not None:
                result = result + (cond - predictions[row]) * spec["scale"]
            if spec["rescaling_scale"] != 0:
                # Not in place: result may still be cond itself.
                result = result * (spec["rescaling_scale"] * (cond.std() / result.std()) + (1 - spec["rescaling_scale"]))
    return result
