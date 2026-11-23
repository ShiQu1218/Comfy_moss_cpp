"""One on-disk merged DiT, using the engine's original CPU merge implementation."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

from filelock import FileLock, Timeout
from safetensors import SafetensorError
import torch
import comfy.model_management as mm
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.lokr import LoKrAdapter

from .lora import adapter_tensors, export_loras

CACHE_ROOT = Path(__file__).resolve().parent / "cache" / "lokr"
HELPER = Path(__file__).resolve().parent / "runtime" / ("moss-lokr-cache.exe" if os.name == "nt" else "moss-lokr-cache")


def split_patches(patches):
    # DoRA depends on preceding patches: bake the entire stack, preserving order.
    dora = any(isinstance(p[1], LoKrAdapter) and p[1].weights[8] is not None
               for stack in patches.values() for p in stack if p[0] != 0)
    if dora:
        for key, stack in patches.items():
            for strength, adapter, model_strength, offset, function in stack:
                if strength == 0:
                    continue
                if model_strength != 1 or function is not None or not key.endswith(".weight"):
                    return None
                if not isinstance(adapter, LoKrAdapter) or adapter_tensors(key[:-7], "weight", adapter, allow_dora=True) is None:
                    return None
                if offset is not None and (not key.endswith(".proj.qkv.weight") or tuple(offset) not in ((0, 0, 3072), (0, 3072, 3072), (0, 6144, 3072))):
                    return None
        return patches, {}
    lokr, remaining = {}, {}
    for key, stack in patches.items():
        for patch in stack:
            strength, adapter, model_strength, offset, function = patch
            if strength == 0:
                continue
            if model_strength != 1 or function is not None:
                return None
            if offset is not None and (not key.endswith(".proj.qkv.weight") or tuple(offset) not in ((0, 0, 3072), (0, 3072, 3072), (0, 6144, 3072))):
                return None
            if isinstance(adapter, LoKrAdapter) and key.endswith(".weight") and adapter_tensors(key[:-7], "weight", adapter) is not None:
                lokr.setdefault(key, []).append(patch)
            elif isinstance(adapter, LoRAAdapter) and key.endswith(".weight") and adapter_tensors(key[:-7], "weight", adapter) is not None:
                block_side_branch = re.fullmatch(
                    r"diffusion_model\.layers\.\d+\.proj\.(qkv|q|k|v|o|ffn_up_gate|ffn_down)\.weight", key)
                global_linear = re.fullmatch(
                    r"diffusion_model\.(?:final_linear_video|shared_block_modulation|video_head_modulation|"
                    r"video_embedder|time_embedding\.[02]|text_embedder\.(?:proj|blocks\.[01]\."
                    r"(?:qkv|proj|ffn_up|ffn_gate|ffn_down)))\.weight", key)
                if not block_side_branch and not global_linear:
                    return None
                remaining.setdefault(key, []).append(patch)
            else:
                # Another merged/nonlinear adapter could depend on the LoKr stack order.
                return None
    # Global linears merge at runtime. Keep a shared target together so it rounds only once.
    for key in lokr.keys() & remaining.keys():
        if not key.startswith("diffusion_model.layers."):
            return None
    return (lokr, remaining) if lokr else None


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(8 << 20):
            mm.throw_exception_if_processing_interrupted()
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(source, loras, helper):
    source = Path(source).resolve()
    stat = source.stat()
    return {"version": 1, "source": str(source), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "fast_rotation": "MOSS_LOKR_FAST_ROTATION" in os.environ and "MOSS_LOKR_CPU" not in os.environ,
            "helper": digest_file(helper),
            "loras": [{"sha256": digest_file(spec["path"]), "strength": spec["strength"]} for spec in loras]}


def build_cache_model(helper, source, output, loras):
    manifest = output.parent / "request.json"
    manifest.write_text(json.dumps({"source": str(Path(source).resolve()), "output": str(output), "loras": loras}), encoding="utf-8")
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([str(helper), str(manifest)], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while True:
                mm.throw_exception_if_processing_interrupted()
                try:
                    process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                log.seek(0, os.SEEK_END)
                log.seek(max(0, log.tell() - 8192))
                raise RuntimeError("LoKr cache helper: " + log.read().decode("utf-8", errors="replace"))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


@contextmanager
def cached_lokr(model, source, temporary, enabled=True):
    split = split_patches(model.patches) if enabled else None
    if split is None:
        yield source, model.patches
        return
    lokr, remaining = split
    # moss-server uses CUDA device 0 and forces all LoRAs to merge on pre-Ampere GPUs.
    if remaining and (not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] < 8):
        yield source, model.patches
        return
    if not HELPER.is_file():
        logging.warning("MossInfer LoKr cache helper not installed; regular loading will be used")
        yield source, model.patches
        return
    with ExitStack() as cache_lock:
        try:
            with tempfile.TemporaryDirectory(prefix="lokr-key-", dir=temporary) as key_directory:
                specs = export_loras(lokr, key_directory, model=model, allow_dora=True)
                key = fingerprint(source, specs, HELPER)
                CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                lock = FileLock(str(CACHE_ROOT / "cache.lock"))
                deadline = time.monotonic() + 300
                while True:
                    mm.throw_exception_if_processing_interrupted()
                    try:
                        cache_lock.enter_context(lock.acquire(timeout=0.2))
                        break
                    except Timeout:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("另一個 ComfyUI 正在使用 LoKr 快取") from None
                model_file = CACHE_ROOT / "merged.safetensors"
                metadata_file = CACHE_ROOT / "manifest.json"
                try:
                    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
                    stat = model_file.stat()
                    hit = metadata["key"] == key and stat.st_size == key["size"] and stat.st_mtime_ns == metadata["output_mtime_ns"]
                except (OSError, ValueError, KeyError, TypeError):
                    hit = False
                if not hit:
                    if shutil.disk_usage(CACHE_ROOT).free < key["size"] + (256 << 20):
                        raise OSError("磁碟空間不足")
                    logging.info("MossInfer LoKr cache MISS: merge once (%d bytes)", key["size"])
                    started = time.perf_counter()
                    with tempfile.TemporaryDirectory(prefix="building-", dir=CACHE_ROOT) as staging:
                        output = Path(staging) / "merged.safetensors"
                        has_dora = any(isinstance(p[1], LoKrAdapter) and p[1].weights[8] is not None
                                       for stack in lokr.values() for p in stack if p[0] != 0)
                        if has_dora:
                            specs = export_loras(lokr, staging, model=model, device=mm.get_torch_device())
                        build_cache_model(HELPER, source, output, specs)
                        stat = Path(source).stat()
                        if stat.st_size != key["size"] or stat.st_mtime_ns != key["mtime_ns"] or output.stat().st_size != key["size"]:
                            raise RuntimeError("建立快取時來源模型已變更，或快取不完整")
                        output.replace(model_file)
                        metadata = {"key": key, "output_mtime_ns": model_file.stat().st_mtime_ns}
                        pending = CACHE_ROOT / "manifest.pending.json"
                        pending.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                        pending.replace(metadata_file)
                    logging.info("MossInfer LoKr cache stored: %.3f s", time.perf_counter() - started)
            if hit:
                logging.info("MossInfer LoKr cache HIT: reuse merged DiT")
        except (OSError, RuntimeError, SafetensorError) as error:
            logging.warning("MossInfer LoKr cache failed; regular loading will be used: %s", error)
        else:
            # Hold the cross-process lock until the engine closes its mapped DiT file.
            yield str(model_file), remaining
            return
    yield source, model.patches
