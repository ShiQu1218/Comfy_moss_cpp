"""Run the modular workflow directly and compare tensor bytes across hardware paths."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfyui", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slg", action="store_true")
    parser.add_argument("--lora", help="ComfyUI LoRA filename; LoKr cache is bypassed")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    if args.size < 16 or args.size % 16:
        parser.error("--size must be a positive multiple of 16")

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(args.comfyui.resolve()))
    sys.argv = [sys.argv[0]]
    import torch
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple
    from nodes import CLIPLoader, CLIPTextEncode, LoraLoaderModelOnly, UNETLoader, VAEDecode, VAELoader

    spec = importlib.util.spec_from_file_location("moss_plugin", root / "__init__.py", submodule_search_locations=[str(root)])
    plugin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plugin
    spec.loader.exec_module(plugin)
    from moss_plugin import sampler

    class ObservedSession(sampler.ComponentSession):
        def __init__(self, *params):
            self.trace = []
            self.trace_seconds = 0.0
            sessions.append(self)
            super().__init__(*params)

        def call(self, method, params=None):
            result = super().call(method, params)
            if method == "run_end":
                self.report = result["report"]
            return result

        def exact_velocities(self, x, value, rows):
            velocities = super().exact_velocities(x, value, rows)
            start = time.perf_counter()
            size = x.numel() * 4
            offsets = [self.offsets["x"], *(self.offsets["out"][row] for row in rows)]
            with memoryview(self.mapping) as view:
                digests = [hashlib.sha256(view[offset:offset + size]).hexdigest() for offset in offsets]
            self.trace.append([value.hex(), rows, digests])
            self.trace_seconds += time.perf_counter() - start
            return velocities

    workflow = json.loads((root / "workflows/modular_api.json").read_text(encoding="utf-8"))
    variants = {"streaming": ("min_memory", False), "resident": ("balanced", False), "graphs": ("balanced", True)}
    model = UNETLoader().load_unet(**workflow["1"]["inputs"])[0]
    if args.lora:
        model = LoraLoaderModelOnly().load_lora_model_only(model, args.lora, 0.7)[0]
    if args.slg:
        model = SkipLayerGuidanceDiTSimple.execute(model, 0.15, 0.7, single_layers="22").result[0]
    clip = CLIPLoader().load_clip(**workflow["2"]["inputs"])[0]
    pos, neg = (CLIPTextEncode().encode(clip, workflow[i]["inputs"]["text"])[0] for i in ("4", "5"))
    vae = VAELoader().load_vae(**workflow["3"]["inputs"])[0]
    latent = {"samples": torch.zeros(1, 128, args.size // 16, args.size // 16)}
    results, sessions = [], []
    baseline = None
    failed = False
    with torch.inference_mode(), patch.object(sampler, "ComponentSession", ObservedSession):
        for run in range(args.runs):
            names = list(variants)
            for name in names[run % len(names):] + names[:run % len(names)]:
                mode, graphs = variants[name]
                start = time.perf_counter()
                result = sampler.MossImageCppSampler.execute(
                    model, pos, neg, latent, args.seed, args.steps, 4, "er_sde", "simple", 1,
                    memory_mode=mode, cuda_graphs=graphs, cache_lokr=False,
                ).result[0]
                session = sessions.pop()
                seconds = time.perf_counter() - start - session.trace_seconds
                pixels = VAEDecode().decode(vae, result)[0].cpu().contiguous().numpy()
                rgb = (pixels * 255).clip(0, 255).astype("uint8")
                sample_bytes = result["samples"].contiguous().numpy().tobytes()
                signature = {"evaluations": session.trace, "latent": hashlib.sha256(sample_bytes).hexdigest(),
                             "vae": hashlib.sha256(pixels).hexdigest(), "rgb": hashlib.sha256(rgb).hexdigest()}
                if baseline is None:
                    baseline = signature
                matches = {key: value == baseline[key] for key, value in signature.items()}
                failed |= not all(matches.values())
                results.append(dict(variant=name, seconds=seconds, matches=matches, signature=signature, report=session.report))
                dit = session.report["plan"]["dit"]
                print(name, round(seconds, 3), matches, session.report["dit_blocks"]["path"],
                      dit["residency"], dit["stream_bytes_per_step"], flush=True)
    medians = {name: statistics.median(r["seconds"] for r in results if r["variant"] == name) for name in variants}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(size=args.size, steps=args.steps, seed=args.seed, slg=args.slg,
                                          lora=args.lora, medians=medians, results=results), indent=2), encoding="utf-8")
    if failed:
        raise SystemExit("Tensor bytes differ from the baseline")


if __name__ == "__main__":
    main()
