"""GPU check: C++ reference latents against ComfyUI's native MOSS sampler on synthetic inputs.

Needs COMFYUI_PATH, a free GPU (close ComfyUI first) and MOSSIMAGE_CPP_EXECUTABLE pointing at a moss-infer.exe whose
sibling moss-server has run_begin reference_latents. Prints relative L2 differences as JSON.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

COMFY = Path(os.environ["COMFYUI_PATH"])
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0]]
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_plugin", root / "__init__.py", submodule_search_locations=[str(root)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["moss_plugin"] = plugin
spec.loader.exec_module(plugin)
from moss_plugin import sampler  # noqa: E402
import torch  # noqa: E402
import nodes  # noqa: E402

TEXT_ENCODER = os.environ.get("MOSS_CHECK_TE", "qwen3.5_9b_int8_convrot.safetensors")
DIT = os.environ.get("MOSS_CHECK_DIT", "moss_image_sft_int8_convrot.safetensors")
SEED = 7
STEPS = int(os.environ.get("MOSS_CHECK_STEPS", 4))
CFG = float(os.environ.get("MOSS_CHECK_CFG", 4.0))


def relative(a, b):
    return float((a - b).norm() / b.norm())


def main():
    model = nodes.UNETLoader().load_unet(DIT, "default")[0]
    generator = torch.Generator().manual_seed(0)
    # Real text-encoder states: random ones are far from their distribution (a few huge channels).
    clip = nodes.CLIPLoader().load_clip(TEXT_ENCODER, "moss_image")[0]
    positive_text = nodes.CLIPTextEncode().encode(clip, "a red ceramic cup on a wooden table, soft light")[0][0][0]
    negative_text = nodes.CLIPTextEncode().encode(clip, "blurry, low quality")[0][0][0]
    del clip
    side = int(os.environ.get("MOSS_CHECK_SIZE", 512)) // 16
    reference = torch.randn(1, 128, side * 3 // 4, side, generator=generator)
    latent = {"samples": torch.zeros(1, 128, side, side)}
    seconds = {}

    def conditioning(text, references):
        metadata = {"reference_latents": [reference], "reference_latents_method": sampler.REFERENCE_METHOD} if references else {}
        return [[text, metadata]]

    def run(engine, references):
        started = time.perf_counter()
        result = sample(engine, references)
        seconds[f"{'cpp' if engine else 'native'}_{'reference' if references else 'plain'}"] = round(time.perf_counter() - started, 2)
        return result

    def sample(engine, references):
        positive, negative = conditioning(positive_text, references), conditioning(negative_text, references)
        if engine:
            # A silent native fallback would make the comparison meaningless.
            with patch.object(sampler, "native_sample", side_effect=AssertionError("the engine path fell back")):
                output = sampler.MossImageCppSampler.execute(model, positive, negative, latent, SEED, STEPS, CFG, "euler",
                                                             "simple", 1.0, cache_lokr=False)
            return output.result[0]["samples"].float()
        return nodes.common_ksampler(model, SEED, STEPS, CFG, "euler", "simple", positive, negative, latent)[0]["samples"].float()

    cpp_plain, native_plain = run(True, False), run(False, False)
    cpp_ref, native_ref = run(True, True), run(False, True)
    print(json.dumps({
        "plain_cpp_vs_native": relative(cpp_plain, native_plain),
        "reference_cpp_vs_native": relative(cpp_ref, native_ref),
        "native_reference_effect": relative(native_ref, native_plain),
        "cpp_reference_effect": relative(cpp_ref, cpp_plain),
        "reference_effect_cpp_vs_native": relative(cpp_ref - cpp_plain, native_ref - native_plain),
        "seconds": seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
