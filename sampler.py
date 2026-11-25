from pathlib import Path
import json
import os
import logging
import tempfile
import time
from types import SimpleNamespace

import torch
import folder_paths
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.model_management as mm
import nodes as comfy_nodes
from comfy_api.latest import io

from .component import ComponentSession
from .nodes import BUNDLE_ROOT, TIERS, MODES, bundle_path, executable_path
from .lora import export_loras
from .guidance import native_guidance
from .lokr_cache import cached_lokr

# The engine's reference layout (Moss-Image-Edit); other methods place the references elsewhere in RoPE.
REFERENCE_METHOD = "index_timestep_zero_text_start"
MAX_REFERENCE_LATENTS = 4
MAX_REFERENCE_TOKENS = 16384
CPP_SAMPLERS = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "dpmpp_2m_sde", "er_sde", "res_multistep"]
KEEP_ENGINE = ["off", "release_vram", "keep_vram"]
# (load key, ComponentSession) kept between runs by keep_engine; ComfyUI runs one prompt at a time.
cached_engine = None
# (executable path, st_size, st_mtime_ns) -> whether moss-server reports run_reference_latents.
server_reference_support = {}


def session_key(executable, load):
    # Same path with new contents (a replaced model) must not reuse an engine that mapped the old weights.
    try:
        stat = os.stat(load["dit_file"])
        identity = [stat.st_size, stat.st_mtime_ns]
    except OSError:
        identity = None
    return json.dumps([str(executable), load, identity], sort_keys=True)


def acquire_session(executable, load, keep_engine):
    global cached_engine
    key = session_key(executable, load)
    if cached_engine is not None:
        (cached_key, session), cached_engine = cached_engine, None
        if keep_engine != "off" and cached_key == key and session.alive():
            return key, session
        session.close()
    return key, ComponentSession(executable, load)


def finish_session(key, session, keep_engine, succeeded):
    global cached_engine
    if keep_engine == "off" or not succeeded:
        session.close()
        return
    try:
        session.finish(keep_engine == "release_vram")
    except Exception as error:
        # The image is already valid; losing the kept engine only costs the next run a reload.
        logging.warning("MossInfer keep_engine: dropping the engine after a failed cleanup: %s", error)
        session.close()
        return
    cached_engine = (key, session)


def drop_cached_session():
    global cached_engine
    if cached_engine is not None:
        cached_engine[1].close()
        cached_engine = None


def model_file(model):
    if model.model.model_config.unet_config.get("image_model") != "moss_image":
        raise ValueError("此節點只支援 MOSS-Image MODEL")
    native_guidance(model.model_options)
    if set(model.object_patches) - {"model_sampling"}:
        raise ValueError("C++ 採樣器尚不支援此 MODEL object patch；請使用原生 KSampler")
    if getattr(model, "weight_wrapper_patches", None):
        raise ValueError("C++ 採樣器尚不支援 MODEL weight wrapper；請使用原生 KSampler")
    factory = model.cached_patcher_init
    if not factory or factory[0] is not comfy.sd.load_diffusion_model:
        raise ValueError("請連接原生 Load Diffusion Model 載入的 MOSS-Image 模型")
    filename, *loader_options = factory[1]
    options = loader_options[0] if loader_options else {}
    if options.get("dtype") is not None or options.get("weight_dtype") is not None or options.get("fp8_optimizations"):
        raise ValueError("C++ 採樣器需要 Load Diffusion Model 的預設 weight_dtype；請使用 default")
    path = Path(os.path.abspath(filename))
    if not path.is_file() or not any(path.is_relative_to(Path(os.path.abspath(root))) for root in folder_paths.get_folder_paths("diffusion_models")):
        raise ValueError("模型檔案不在 ComfyUI diffusion_models 搜尋目錄內")
    return str(path)


def plain_conditioning(conditioning):
    if len(conditioning) != 1:
        raise ValueError("目前支援單組 conditioning；多區域／時間排程請用原生 KSampler")
    embeddings, metadata = conditioning[0]
    if embeddings.ndim != 3 or embeddings.shape[0] != 1 or embeddings.shape[2] != 4096 or not 1 <= embeddings.shape[1] <= 4096:
        raise ValueError("需要原生 MOSS-Image CLIP Text Encode 的 [1, tokens, 4096] conditioning")
    if set(metadata) - {"pooled_output", "attention_mask", "reference_latents", "reference_latents_method"}:
        raise ValueError("C++ 採樣器尚不支援帶有 ControlNet、參考圖或其他額外設定的 conditioning")
    mask = metadata.get("attention_mask")
    if mask is not None and (mask.shape != embeddings.shape[:2] or not torch.all(mask == 1)):
        raise ValueError("目前不支援含有 padding 的 conditioning attention mask")
    return embeddings


def engine_references(positive, negative):
    """The reference latents the C++ engine runs as ComfyUI would, [] without any, None when only ComfyUI can."""
    metadata = [entry[1] for entry in positive + negative]
    found = [entry.get("reference_latents") or [] for entry in metadata]
    if not any(found):
        return []
    references = found[0]
    # The engine gives every row the same references; ComfyUI gives each conditioning its own.
    same = all(len(other) == len(references) and all(a is b or (a.shape == b.shape and torch.equal(a, b))
                                                    for a, b in zip(other, references)) for other in found[1:])
    if not same or any(entry.get("reference_latents_method") != REFERENCE_METHOD for entry in metadata):
        return None
    if len(references) > MAX_REFERENCE_LATENTS or any(latent.ndim != 4 or latent.shape[:2] != (1, 128) for latent in references):
        return None
    if sum(latent.shape[-2] * latent.shape[-1] for latent in references) > MAX_REFERENCE_TOKENS:
        return None
    return references


def native_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, reason):
    logging.warning("MossInfer: %s; using the native sampler", reason)
    return io.NodeOutput(comfy_nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                                                     positive, negative, latent_image, denoise=denoise)[0])


def component_metadata_file(kind, prefix):
    # The component API checks TE/VAE headers even when it only evaluates the DiT.
    names = [name for name in folder_paths.get_filename_list(kind) if Path(name).name.startswith(prefix)]
    if not names:
        raise FileNotFoundError(f"MossInfer 需要 {prefix} 的本機元件標頭；請參閱 README")
    return folder_paths.get_full_path_or_raise(kind, names[0])


def sampling_sigmas(sampling, scheduler, steps, denoise):
    count = int(steps / denoise) if denoise < 1 else steps
    if scheduler == "kl_optimal" and count == 1:
        return torch.tensor([float(sampling.sigma_max), 0.0], dtype=torch.float32)
    sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, count)[-(steps + 1):]
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any() or (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError(f"排程器 {scheduler} 產生無效的 sigma 序列")
    # Discrete schedules can repeat sigmas at low denoise; zero-length steps break SDE solvers.
    sigmas = torch.unique_consecutive(sigmas)
    if len(sigmas) < 2:
        raise ValueError(f"排程器 {scheduler} 未產生可執行的採樣步驟")
    return sigmas


class MossImageCppSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bundles = sorted(p.name for p in BUNDLE_ROOT.iterdir() if p.is_dir()) if BUNDLE_ROOT.is_dir() else []
        return io.Schema(node_id="MossImageCppSampler", display_name="KSampler (MOSS Image C++)", category="MOSS Image",
            description="替換 MOSS-Image 的 KSampler；沿用原生 MODEL、Load LoRA、SLG、CONDITIONING、LATENT 與 VAE Decode。",
            inputs=[
                io.Model.Input("model"), io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Int.Input("steps", default=50, min=1, max=1000),
                io.Float.Input("cfg", default=4.0, min=0, max=100, step=0.1),
                io.Combo.Input("sampler_name", options=CPP_SAMPLERS, default="er_sde"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Float.Input("denoise", default=1.0, min=0, max=1, step=0.01),
                io.Combo.Input("bundle", options=bundles or ["MOSS-Image-SFT"], default="MOSS-Image-SFT", optional=True, advanced=True),
                io.Combo.Input("tier", options=TIERS, default="high", optional=True, advanced=True),
                io.Combo.Input("memory_mode", options=MODES, default="min_memory", optional=True, advanced=True),
                io.Boolean.Input("cuda_graphs", default=False, optional=True, advanced=True,
                                 tooltip="DiT 權重全數常駐且推論形狀重複時使用 CUDA Graph；可搭配 balanced。"),
                io.Boolean.Input("cache_lokr", default=True, optional=True, advanced=True,
                                 tooltip="快取最近一組 LoKr 合併結果至磁碟；首次仍需合併，約占一份 DiT 的空間。"),
                io.Boolean.Input("forecast_denoise", default=False, optional=True, advanced=True,
                                 tooltip="實驗性加速：預測部分去噪速度張量以省略 DiT 推論，可能改變圖片細節。"),
                io.Combo.Input("keep_engine", options=KEEP_ENGINE, default="off", optional=True, advanced=True,
                               tooltip="在兩張圖之間保留引擎程序，省去每張重新載入 DiT（輸出位元相同）。"
                                       "release_vram：每張後釋放 VRAM；keep_vram：DiT 留在 VRAM，最快但 ComfyUI 無法回收這塊記憶體。"),
            ], outputs=[io.Latent.Output()])

    @classmethod
    def execute(cls, model, positive, negative, latent_image, seed, steps, cfg, sampler_name, scheduler, denoise,
                bundle="MOSS-Image-SFT", tier="high", memory_mode="min_memory", cache_lokr=True, forecast_denoise=False,
                cuda_graphs=False, keep_engine="off"):
        if torch.version.hip or not torch.cuda.is_available():
            drop_cached_session()
            return native_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                                 "needs an NVIDIA CUDA GPU")
        references = engine_references(positive, negative)
        if references is None:
            drop_cached_session()  # a kept engine's VRAM would compete with the native model
            return native_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                                 f"reference latents need the same set on every conditioning and method {REFERENCE_METHOD}")
        path = model_file(model)
        pos, neg = plain_conditioning(positive), plain_conditioning(negative)
        latent_image = latent_image.copy()
        latent_image["samples"] = comfy.sample.fix_empty_latent_channels(
            model, latent_image["samples"], latent_image.get("downscale_ratio_spacial"),
            latent_image.get("downscale_ratio_temporal"))
        latent_image.pop("downscale_ratio_spacial", None)
        latent_image.pop("downscale_ratio_temporal", None)
        latent = latent_image["samples"]
        if latent.ndim != 4 or latent.shape[:2] != (1, 128):
            raise ValueError("目前支援 batch 1 的 MOSS-Image LATENT [1,128,h,w]")
        if set(latent_image) - {"samples", "downscale_ratio_spacial", "downscale_ratio_temporal"}:
            raise ValueError("目前不支援 noise mask、batch_index 或額外 latent 設定")
        if (tier not in TIERS or memory_mode not in MODES or sampler_name not in CPP_SAMPLERS
                or scheduler not in comfy.samplers.KSampler.SCHEDULERS or keep_engine not in KEEP_ENGINE):
            raise ValueError("不支援的引擎、採樣器或排程器設定")
        if not 1 <= steps <= 1000 or not 0 <= denoise <= 1:
            raise ValueError("steps 須介於 1 和 1000，denoise 須介於 0 和 1")
        if denoise == 0:
            return io.NodeOutput(latent_image.copy())
        sampling = model.get_model_object("model_sampling")
        sigmas = sampling_sigmas(sampling, scheduler, steps, denoise)
        load = {"model": str(bundle_path(bundle)), "dit_file": path, "tier": tier, "mode": memory_mode,
                "text_encoder_file": component_metadata_file("text_encoders", "qwen3.5_9b"),
                "vae_file": component_metadata_file("vae", "flux2")}
        if cuda_graphs:
            load["block_path"] = "fused_graphs"
        executable = executable_path().with_name("moss-server.exe" if os.name == "nt" else "moss-server")
        if not executable.is_file():
            raise FileNotFoundError(f"找不到 {executable}")
        stat = os.stat(executable)
        server_key = (str(executable), stat.st_size, stat.st_mtime_ns)
        if references and server_reference_support.get(server_key) is False:
            drop_cached_session()
            return native_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                 denoise, "this moss-server has no run_begin reference_latents")
        progress = comfy.utils.ProgressBar(len(sigmas) - 1)
        progress.update_absolute(0)
        started = time.perf_counter()
        mm.unload_all_models()
        mm.soft_empty_cache()
        logging.info("MossInfer preparation unload: %.3f s", time.perf_counter() - started)
        device = mm.get_torch_device()
        noise = comfy.sample.prepare_noise(latent, seed).to(device)
        x = sampling.noise_scaling(sigmas[0].to(device), noise, latent.to(device), bool(sigmas[0] >= sampling.sigma_max))
        with tempfile.TemporaryDirectory(prefix="moss-lora-", dir=folder_paths.get_temp_directory()) as temporary, \
                cached_lokr(model, path, temporary, enabled=cache_lokr and tier != "turbo") as (dit_file, patches):
            load["dit_file"] = dit_file
            started = time.perf_counter()
            loras = export_loras(patches, temporary, model=model)
            logging.info("MossInfer preparation LoRA export: %.3f s (%d files)", time.perf_counter() - started, len(loras))
            started = time.perf_counter()
            key, session = acquire_session(executable, load, keep_engine)
            logging.info("MossInfer preparation server: %.3f s", time.perf_counter() - started)
            server_reference_support[server_key] = bool(session.reference_latents)
            if references and not session.reference_latents:
                session.close()
                return native_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                     denoise, "this moss-server has no run_begin reference_latents")
            succeeded = False
            try:
                logging.info("MossInfer preparing DiT and merging adapters (%d LoRA files)...", len(loras))
                started = time.perf_counter()
                session.begin(pos, neg, latent.shape[-1] * 16, latent.shape[-2] * 16, sigmas, cfg, loras=loras,
                              guidance=native_guidance(model.model_options), forecast_denoise=forecast_denoise,
                              references=references)
                logging.info("MossInfer preparation run_begin: %.3f s", time.perf_counter() - started)
                first_evaluation = True
                def evaluate(x, sigma, **kwargs):
                    nonlocal first_evaluation
                    started = time.perf_counter()
                    result = session.denoise(x, sigma)
                    if first_evaluation:
                        logging.info("MossInfer preparation first evaluation: %.3f s", time.perf_counter() - started)
                        first_evaluation = False
                    return result
                evaluate.inner_model = SimpleNamespace(model_patcher=model, inner_model=SimpleNamespace(model_sampling=sampling))
                def callback(state):
                    mm.throw_exception_if_processing_interrupted()
                    progress.update_absolute(state["i"] + 1)
                sampler = comfy.samplers.ksampler(sampler_name)
                samples = sampler.sampler_function(evaluate, x, sigmas.to(device), extra_args={"seed": seed},
                                                   callback=callback, disable=True, **sampler.extra_options)
                report = session.call("run_end")["report"]
                if forecast_denoise:
                    logging.info("MossInfer denoise forecast: %d/%d evaluations", session.forecasted_calls, session.denoise_calls)
                if loras:
                    applied = report["loras"]
                    if len(applied) != len(loras) or any(item["unknown"] or not item["adapters"] for item in applied):
                        raise RuntimeError("C++ 引擎未完整套用 LoRA，請查看模型相容性")
                    logging.info("MossInfer LoRA report: %s", applied)
                result = latent_image.copy()
                result["samples"] = sampling.inverse_noise_scaling(sigmas[-1].to(device), samples).cpu()
                if not torch.isfinite(result["samples"]).all():
                    raise RuntimeError("採樣結果含有 NaN 或無限值")
                succeeded = True
                return io.NodeOutput(result)
            finally:
                finish_session(key, session, keep_engine, succeeded)
