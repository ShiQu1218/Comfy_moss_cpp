"""Export native ComfyUI LoRA patches without changing the base model."""
import math
import logging
import threading
from pathlib import Path

import torch
import comfy.utils
import comfy.lora
import comfy.model_patcher
from comfy.weight_adapter.base import WeightAdapterBase
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.loha import LoHaAdapter
from comfy.weight_adapter.lokr import LoKrAdapter


def adapter_tensors(stem, kind, adapter, allow_dora=False):
    if isinstance(adapter, LoRAAdapter):
        up, down, alpha, mid, dora, reshape = adapter.weights
        if kind != "weight" or mid is not None or dora is not None or reshape is not None or up.ndim != 2 or down.ndim != 2:
            return None
        return {stem + ".lora_up.weight": up, stem + ".lora_down.weight": down,
                stem + ".alpha": torch.tensor(down.shape[0] if alpha is None else alpha, dtype=torch.float32)}
    if isinstance(adapter, LoHaAdapter):
        w1a, w1b, alpha, w2a, w2b, t1, t2, dora = adapter.weights
        if kind != "weight" or any(value is not None for value in (t1, t2, dora)) or any(value.ndim != 2 for value in (w1a, w1b, w2a, w2b)):
            return None
        return {stem + ".hada_w1_a": w1a, stem + ".hada_w1_b": w1b,
                stem + ".hada_w2_a": w2a, stem + ".hada_w2_b": w2b,
                stem + ".alpha": torch.tensor(w1b.shape[0] if alpha is None else alpha, dtype=torch.float32)}
    if isinstance(adapter, LoKrAdapter):
        w1, w2, alpha, w1a, w1b, w2a, w2b, t2, dora = adapter.weights
        if kind != "weight" or t2 is not None or (dora is not None and not allow_dora) or any(value is not None and value.ndim != 2 for value in (w1, w2, w1a, w1b, w2a, w2b)):
            return None
        rank = w2b.shape[0] if w2 is None else w1b.shape[0] if w1 is None else None
        # Native ComfyUI ignores alpha when neither Kronecker factor is decomposed.
        alpha = 1 if rank is None else rank if alpha is None else alpha
        values = {"lokr_w1": w1, "lokr_w2": w2, "lokr_w1_a": w1a, "lokr_w1_b": w1b,
                  "lokr_w2_a": w2a, "lokr_w2_b": w2b, "alpha": torch.tensor(alpha, dtype=torch.float32)}
        if allow_dora:
            values["dora_scale"] = dora
        return {stem + "." + name: value for name, value in values.items() if value is not None}
    if isinstance(adapter, tuple) and len(adapter) == 2 and adapter[0] == "diff":
        return {stem + (".diff" if kind == "weight" else ".diff_b"): adapter[1][0]}
    if isinstance(adapter, WeightAdapterBase):
        return None
    raise ValueError(f"不支援的 LoRA adapter：{type(adapter).__name__}")


def full_delta(model, key, stack, device="cpu"):
    if model is None:
        raise ValueError(f"此 LoRA 格式需要原始 MODEL 權重才能轉換：{key}")
    weight, _, convert = comfy.model_patcher.get_key_weight(model.model, key)
    if key in model.backup:
        weight = model.backup[key].weight
    if key in model.hook_backup:
        weight = model.hook_backup[key][0]
    base = weight.detach().to("cpu")
    if convert is not None:
        base = convert(base, inplace=False)
    base = base.to(device=device, dtype=torch.float32).clone()
    failures = []
    owner = threading.get_ident()
    class PatchErrors(logging.Handler):
        def emit(self, record):
            if record.thread == owner and record.levelno >= logging.WARNING:
                failures.append(record.getMessage())
    handler = PatchErrors()
    logging.getLogger().addHandler(handler)
    try:
        patched = comfy.lora.calculate_weight(stack, base.clone(), key)
    finally:
        logging.getLogger().removeHandler(handler)
    if failures or patched.shape != base.shape or not torch.isfinite(patched).all():
        raise ValueError(f"無法保留此 LoRA 的原生計算結果：{key} {failures}")
    return (patched - base).cpu()


def export_loras(patches, directory, model=None, allow_dora=False, device="cpu"):
    groups = []  # ordered [((index, strength), tensors)]
    result = []
    merged = {}
    merged_bytes = 0

    def flush_merged():
        if merged:
            path = Path(directory) / f"merged_{len(result)}.safetensors"
            comfy.utils.save_torch_file(merged, str(path))
            result.append({"path": str(path), "strength": 1.0})
            merged.clear()

    for key, stack in patches.items():
        if not key.startswith("diffusion_model.") or not key.endswith((".weight", ".bias")):
            raise ValueError(f"不支援的 LoRA 目標：{key}")
        pending = []
        materialize = False
        index = 0  # counts only applied patches, so a zero-strength partial LoRA cannot shift grouping
        for strength, adapter, strength_model, offset, function in stack:
            if strength_model != 1.0 or function is not None or not math.isfinite(strength):
                raise ValueError(f"無法轉送此 MODEL patch 的縮放或轉換：{key}")
            if strength == 0:
                continue
            target = key
            if offset is not None:
                # ComfyUI maps separate Q/K/V adapters onto slices of its fused QKV weight.
                dimension, start, length = offset
                if not key.endswith(".proj.qkv.weight") or dimension != 0 or length != 3072 or start not in (0, 3072, 6144):
                    materialize = True
                else:
                    target = key.replace(".qkv.weight", f".{('q', 'k', 'v')[start // 3072]}.weight")
            stem, kind = target.rsplit(".", 1)
            tensors = adapter_tensors(stem, kind, adapter, allow_dora=allow_dora)
            materialize |= tensors is None
            pending.append((target, index, float(strength), tensors))
            index += 1
        if materialize:
            stem, kind = key.rsplit(".", 1)
            delta = full_delta(model, key, stack, device=device)
            size = delta.numel() * delta.element_size()
            # Independent target deltas share files; bound retained CPU tensors while exporting.
            if merged_bytes + size > 512 << 20:
                flush_merged()
                merged_bytes = 0
            merged[stem + (".diff" if kind == "weight" else ".diff_b")] = delta
            merged_bytes += size
            logging.info("MossInfer materialized native adapter stack: %s (%d bytes)", key, delta.numel() * delta.element_size())
            continue
        last = {}  # target -> position of the group holding its previous patch
        for target, index, strength, tensors in pending:
            # A target's next patch must land in a later file than its previous one: files apply in order.
            position = next((i for i in range(last.get(target, -1) + 1, len(groups))
                             if groups[i][0] == (index, strength) and not groups[i][1].keys() & tensors.keys()), None)
            if position is None:
                position = len(groups)
                groups.append(((index, strength), {}))
            groups[position][1].update(tensors)
            last[target] = position
    flush_merged()
    for (index, strength), tensors in groups:
        path = Path(directory) / f"lora_{len(result)}.safetensors"
        # Clones prevent shared QKV factors from becoming aliased safetensors entries.
        tensors = {key: value.detach().to("cpu").contiguous().clone() for key, value in tensors.items()}
        comfy.utils.save_torch_file(tensors, str(path))
        result.append({"path": str(path), "strength": strength})
    return result
