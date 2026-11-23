import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch

import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import io


BUNDLE_ROOT = Path(folder_paths.models_dir) / "mossimage"
TIERS = ["high", "reference", "turbo"]
MODES = ["min_memory", "balanced", "max_speed"]
SAMPLERS = ["euler", "ab2", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep"]
SCHEDULERS = ["moss", "simple", "normal", "karras", "exponential", "beta", "sgm_uniform"]


def bundle_path(name):
    root = BUNDLE_ROOT.resolve()
    path = (root / name).resolve()
    if not name or not path.is_relative_to(root) or path == root:
        raise ValueError("模型設定目錄必須位於 ComfyUI/models/mossimage 內")
    for component in ("transformer", "text_encoder", "vae"):
        if not (path / component / "config.json").is_file():
            raise FileNotFoundError(f"缺少 {component}/config.json：{path}")
    if not (path / "tokenizer" / "tokenizer.json").is_file():
        raise FileNotFoundError(f"缺少 tokenizer/tokenizer.json：{path}")
    return path


def executable_path():
    configured = os.environ.get("MOSSIMAGE_CPP_EXECUTABLE")
    if configured:
        path = Path(configured).expanduser().resolve()
    else:
        runtime = Path(__file__).resolve().parent / "runtime"
        name = "moss-infer.exe" if os.name == "nt" else "moss-infer"
        candidates = sorted(runtime.glob(f"*/bin/{name}"))
        if len(candidates) != 1:
            raise FileNotFoundError("請將一份 MossInfer 套件解壓至節點的 runtime/，或設定 MOSSIMAGE_CPP_EXECUTABLE")
        path = candidates[0]
    if not path.is_file():
        raise FileNotFoundError(f"找不到 moss-infer：{path}")
    return path


class MossImageCpp(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bundles = sorted(p.name for p in BUNDLE_ROOT.iterdir() if p.is_dir()) if BUNDLE_ROOT.is_dir() else []
        return io.Schema(
            node_id="MossImageCpp", display_name="MOSS Image C++ (Legacy All-in-One)",
            category="MOSS Image", description="保留舊 workflow 的整合節點；現有流程請改用 KSampler (MOSS Image C++)。",
            inputs=[
                io.Combo.Input("bundle", options=bundles or ["MOSS-Image-SFT"]),
                io.Combo.Input("dit", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input("text_encoder", options=folder_paths.get_filename_list("text_encoders")),
                io.Combo.Input("vae", options=folder_paths.get_filename_list("vae")),
                io.String.Input("prompt", multiline=True, default="A red apple on a wooden table, natural light, photography"),
                io.String.Input("negative_prompt", multiline=True, default=""),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True),
                io.Int.Input("width", default=1024, min=16, max=4096, step=16),
                io.Int.Input("height", default=1024, min=16, max=4096, step=16),
                io.Int.Input("steps", default=50, min=1, max=1000),
                io.Float.Input("cfg", default=3.0, min=0, max=100, step=0.1),
                io.Float.Input("shift", default=5.0, min=0.01, max=100, step=0.1),
                io.Combo.Input("tier", options=TIERS),
                io.Combo.Input("memory_mode", options=MODES),
                io.Combo.Input("sampler", options=SAMPLERS, default="euler", optional=True),
                io.Combo.Input("scheduler", options=SCHEDULERS, default="moss", optional=True),
            ], outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, bundle, dit, text_encoder, vae, prompt, negative_prompt,
                seed, width, height, steps, cfg, shift, tier, memory_mode, sampler="euler", scheduler="moss"):
        if tier not in TIERS or memory_mode not in MODES:
            raise ValueError("不支援的品質或記憶體模式")
        if sampler not in SAMPLERS or scheduler not in SCHEDULERS:
            raise ValueError("不支援的採樣器或排程器")
        if width % 16 or height % 16 or not 16 <= width <= 4096 or not 16 <= height <= 4096 or width * height > 4200000:
            raise ValueError("長寬必須為 16 的倍數，且總像素不得超過 4200000")
        executable = executable_path()
        model = bundle_path(bundle)
        components = [folder_paths.get_full_path_or_raise(kind, name) for kind, name in (
            ("diffusion_models", dit), ("text_encoders", text_encoder), ("vae", vae))]
        model_management.throw_exception_if_processing_interrupted()
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        # shortcut: 每次重新載入模型；需要連續大量生成時再改用常駐 moss-server。
        with tempfile.TemporaryDirectory(prefix="mossimage-", dir=folder_paths.get_temp_directory()) as temporary:
            output = Path(temporary) / "image.png"
            report = Path(temporary) / "report.json"
            command = [str(executable), "generate", "--model", str(model),
                       "--dit", components[0], "--te", components[1], "--vae", components[2],
                       "--prompt", prompt, "--negative", negative_prompt,
                       "--seed", str(seed), "--width", str(width), "--height", str(height),
                       "--steps", str(steps), "--cfg", str(cfg), "--shift", str(shift),
                       "--tier", tier, "--mode", memory_mode,
                       "--sampler", sampler, "--scheduler", scheduler,
                       "--json-report", str(report), "-o", str(output)]
            with tempfile.TemporaryFile() as log:
                process = subprocess.Popen(command, stdout=log, stderr=log,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                try:
                    while True:
                        model_management.throw_exception_if_processing_interrupted()
                        try:
                            process.wait(timeout=0.2)
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    if process.returncode:
                        log.seek(0, os.SEEK_END)
                        log.seek(max(0, log.tell() - 8192))
                        raise RuntimeError(f"MossInfer 結束碼 {process.returncode}\n{log.read().decode('utf-8', errors='replace')}")
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
            timings = json.loads(report.read_text(encoding="utf-8"))["timings_s"]
            logging.info("MossInfer timings_s: %s", json.dumps(timings))
            with Image.open(output) as image:
                pixels = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            return io.NodeOutput(torch.from_numpy(pixels).unsqueeze(0))
