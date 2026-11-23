"""Compare the C++ node and native ComfyUI with identical requested settings."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import uuid

import websocket


ROOT = Path(__file__).resolve().parents[1]


def request(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}), timeout=30) as response:
        content = response.read()
        return json.loads(content) if content else None


def graphs():
    cpp = json.loads((ROOT / "workflows/txt2img_api.json").read_text(encoding="utf-8"))
    settings = cpp["1"]["inputs"]
    settings.update(width=1024, height=1024, steps=50, cfg=4.0, sampler="er_sde", scheduler="simple")
    cpp["2"]["inputs"]["filename_prefix"] = "moss_benchmark_cpp"
    native = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": settings["dit"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": settings["text_encoder"], "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": settings["vae"]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": settings["prompt"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": settings["negative_prompt"]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "10": {"class_type": "SkipLayerGuidanceDiTSimple", "inputs": {"model": ["1", 0], "double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0], "seed": settings["seed"], "steps": 50, "cfg": 4.0, "sampler_name": "er_sde", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "moss_benchmark_native"}},
    }
    return {"cpp": cpp, "native": native}


def run(url, name, graph, iteration, directory, unload=True):
    queue = request(url + "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("ComfyUI queue is busy")
    if unload:
        request(url + "/free", {"unload_models": True, "free_memory": True})
        time.sleep(2)
    client = str(uuid.uuid4())
    ws = websocket.create_connection(url.replace("http", "ws", 1) + "/ws?clientId=" + client, timeout=30)
    try:
        started = time.perf_counter()
        submitted = request(url + "/prompt", {"prompt": graph, "client_id": client})
        if submitted.get("node_errors"):
            raise RuntimeError(submitted)
        prompt_id = submitted["prompt_id"]
        events = []
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            try:
                message = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not isinstance(message, str):
                continue
            event = json.loads(message)
            if event.get("data", {}).get("prompt_id") != prompt_id:
                continue
            event["elapsed_s"] = time.perf_counter() - started
            events.append(event)
            if event["type"] in ("execution_error", "execution_interrupted"):
                raise RuntimeError(event)
            if event["type"] == "execution_success":
                break
        else:
            raise TimeoutError(prompt_id)
        history = request(url + "/history/" + prompt_id)[prompt_id]
        if history["status"]["status_str"] != "success":
            raise RuntimeError(history["status"])
        cached = [node for e in events if e["type"] == "execution_cached" for node in e["data"]["nodes"]]
        if (unload and cached) or any(graph[node]["class_type"] in ("KSampler", "MossImageCpp", "MossImageCppSampler", "SaveImage") for node in cached):
            raise RuntimeError("Cached nodes would invalidate the timing comparison")
        result = {"name": name, "iteration": iteration, "wall_s": events[-1]["elapsed_s"], "prompt_id": prompt_id, "events": events, "history": history}
        (directory / f"{name}_{iteration}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name} #{iteration}: {result['wall_s']:.3f} s", flush=True)
        return result
    finally:
        ws.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8189")
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    directory = ROOT / "test_results/benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    workflows = graphs()
    for name, graph in workflows.items():
        (ROOT / f"workflows/benchmark_{name}_api.json").write_text(json.dumps(graph, indent=2), encoding="utf-8")
    results = []
    for iteration in range(1, args.runs + 1):
        for name, graph in workflows.items():
            results.append(run(args.url, name, graph, iteration, directory))
    for name in ("native", "cpp"):
        graph = workflows[name]
        graph["7" if name == "native" else "1"]["inputs"]["seed"] = 43
        results.append(run(args.url, name + "_warm", graph, 1, directory, unload=False))
    (directory / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
