"""Submit the example to a running ComfyUI; fail on validation/execution errors."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def request(url, data=None):
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}), timeout=30) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    graph = json.loads((root / "workflows/txt2img_api.json").read_text(encoding="utf-8"))
    response = request(args.url + "/prompt", {"prompt": graph})
    if response.get("node_errors"):
        raise RuntimeError(response)
    prompt_id = response["prompt_id"]
    print(f"prompt_id={prompt_id}", flush=True)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        history = request(args.url + "/history/" + prompt_id)
        if prompt_id in history:
            result = history[prompt_id]
            destination = root / "test_results"
            destination.mkdir(exist_ok=True)
            (destination / "workflow_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            if result["status"]["status_str"] != "success" or not result.get("outputs", {}).get("2", {}).get("images"):
                raise RuntimeError(json.dumps(result["status"], ensure_ascii=False))
            print(json.dumps(result["outputs"], ensure_ascii=False), flush=True)
            return
        time.sleep(1)
    raise TimeoutError(f"工作尚未完成，請至 ComfyUI 查看：{prompt_id}")


if __name__ == "__main__":
    main()
