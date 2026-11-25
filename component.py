"""Local MossInfer component protocol; tensors stay in shared memory."""
import json
import mmap
import os
from pathlib import Path
import queue
import struct
import subprocess
import tempfile
import threading
import time

import numpy as np
import torch
import comfy.model_management as mm
from .guidance import prediction_rows, combine_predictions


def attach_region(region, pid):
    size = region["bytes"]
    if not 4096 <= size <= (1 << 34) + 4096:
        raise ValueError("Invalid MossInfer shared-memory size")
    name = region["name"]
    if os.name == "nt":
        if not name.startswith(f"Local\\mossinfer-{pid}-"):
            raise ValueError("Invalid MossInfer mapping name")
        mapping = mmap.mmap(-1, size, tagname=name)
    else:
        if not name.startswith(f"/mossinfer-{pid}-") or "/" in name[1:]:
            raise ValueError("Invalid MossInfer mapping name")
        descriptor = os.open("/dev/shm/" + name[1:], os.O_RDWR)
        try:
            mapping = mmap.mmap(descriptor, size)
        finally:
            os.close(descriptor)
    magic, version, owner, token, declared, offset = struct.unpack("<8sII16sQQ", mapping[:48])
    if (magic != b"MOSSSHM\0" or version != 1 or owner != pid or
            token.hex() != region["token"] or declared != size or offset != 4096):
        mapping.close()
        raise ValueError("MossInfer shared-memory header mismatch")
    return mapping


class ComponentSession:
    rows_per_call = 8  # session_open's max_rows
    reference_latents = False  # the server's run_begin takes reference_latents

    def __init__(self, executable, load):
        self.log = tempfile.TemporaryFile()
        self.messages = queue.Queue()
        self.sequence = 0
        self.mapping = None
        self.process = None
        try:
            self.process = subprocess.Popen([str(executable)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.reader = threading.Thread(target=self.read_messages, daemon=True)
            self.reader.start()
            hello = self.call("hello", {"protocol_version": 2})
            self.reference_latents = bool(hello.get("capabilities", {}).get("run_reference_latents"))
            self.call("load", load)
            self.call("session_open", {"max_rows": 8})
        except BaseException:
            self.close()
            raise

    def read_messages(self):
        for line in self.process.stdout:
            self.messages.put(line)
        self.messages.put(None)

    def call(self, method, params=None):
        self.sequence += 1
        self.process.stdin.write((json.dumps({"id": self.sequence, "method": method, "params": params or {}}) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            mm.throw_exception_if_processing_interrupted()
            try:
                line = self.messages.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                self.log.seek(0, os.SEEK_END)
                self.log.seek(max(0, self.log.tell() - 8192))
                raise RuntimeError("MossInfer server exited: " + self.log.read().decode("utf-8", errors="replace"))
            response = json.loads(line)
            if "event" in response:
                continue
            if response.get("id") != self.sequence:
                raise RuntimeError("MossInfer response id mismatch")
            if "error" in response:
                raise RuntimeError(f"MossInfer {method}: {response['error']['message']}")
            return response["result"]
        raise TimeoutError(f"MossInfer {method} timed out")

    def import_conditioning(self, embeddings):
        array = embeddings[0].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        region = self.call("shm_alloc", {"bytes": array.nbytes})["shm"]
        mapping = attach_region(region, self.process.pid)
        try:
            mapping[4096:4096 + array.nbytes] = array.tobytes()
            conditioning = self.call("conditioning_import", {"shm": region["name"], "dtype": "f32", "shape": list(array.shape)})["id"]
        finally:
            mapping.close()
        # The import copied the payload; a kept engine would otherwise hit the session's scratch-memory cap.
        self.call("shm_free", {"name": region["name"]})
        return conditioning

    def import_reference(self, latent):
        # The server copies the latents at run_begin; the region only has to live until then.
        array = latent[0].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        region = self.call("shm_alloc", {"bytes": array.nbytes})["shm"]
        mapping = attach_region(region, self.process.pid)
        try:
            mapping[4096:4096 + array.nbytes] = array.tobytes()
        finally:
            mapping.close()
        return {"shm": region["name"], "shape": list(array.shape)}

    def begin(self, positive, negative, width, height, sigmas, cfg, loras=None, guidance=None, forecast_denoise=False,
              references=None):
        ids = [self.import_conditioning(c) for c in (positive, negative)]
        self.conditionings = ids
        self.guidance = guidance or {"simple": None, "post": []}
        self.cfg = cfg
        rows = [{"conditioning": i} for i in ids]
        if self.guidance["simple"] is not None:
            rows.append({"conditioning": ids[1], "skip_layers": self.guidance["simple"]["layers"]})
        rows.extend({"conditioning": ids[0], "skip_layers": spec["layers"]} for spec in self.guidance["post"])
        if len(rows) > 8:
            raise ValueError("跳層引導組合超過引擎的 8 個 prediction rows 上限")
        params = {"width": width, "height": height, "rows": rows, "guidance": "host", "output": "rows",
                  "sigmas": sigmas.tolist(), "exact_sigma": False, "loras": loras or []}
        if references:
            params["reference_latents"] = [self.import_reference(latent) for latent in references]
        # The engine's reference path evaluates at most 2 rows per forward.
        self.rows_per_call = 2 if references else len(rows)
        result = self.call("run_begin", params)
        # run_begin copied the references; free their scratch regions as the conditioning ones.
        for reference in params.get("reference_latents", []):
            self.call("shm_free", {"name": reference["shm"]})
        self.mapping = attach_region(result["shm"], self.process.pid)
        self.offsets = result["tensors"]
        self.forecast_denoise = forecast_denoise and len(sigmas) >= 8
        if self.forecast_denoise:
            self.forecast_tail = float(sigmas[-3])
            self.forecast_rows = []
            self.velocity_history = []
        self.denoise_calls = 0
        self.forecasted_calls = 0

    def exact_velocities(self, x, value, rows):
        array = x.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        start = self.offsets["x"]
        for offset in [start] + [self.offsets["out"][row] for row in rows]:
            if not isinstance(offset, int) or offset < 4096 or offset + array.nbytes > len(self.mapping):
                raise ValueError("MossInfer tensor outside shared-memory region")
        np.copyto(np.ndarray(array.shape, dtype=np.float32, buffer=self.mapping, offset=start), array)
        for first in range(0, len(rows), self.rows_per_call):
            mask = sum(1 << row for row in rows[first:first + self.rows_per_call])
            evaluated = self.call("denoise_step", {"sigma": value, "rows": mask})
            if evaluated["rows"] & mask != mask:
                raise RuntimeError("MossInfer 未完成指定的 guidance rows")
        velocities = []
        for row in rows:
            out = self.offsets["out"][row]
            velocity = np.ndarray(array.shape, dtype=np.float32, buffer=self.mapping, offset=out).copy()
            velocities.append(torch.from_numpy(velocity).to(x.device))
        return velocities

    def denoise(self, x, sigma):
        value = float(sigma[0])
        rows = prediction_rows(self.guidance, value, self.cfg)
        self.denoise_calls += 1
        velocities = None
        if self.forecast_denoise and value > self.forecast_tail:
            if rows != self.forecast_rows:
                self.velocity_history.clear()
                self.forecast_rows = rows
            if self.denoise_calls > 4 and len(self.velocity_history) == 2:
                (earlier, older), (latest, newer) = self.velocity_history
                interval, distance = earlier - latest, latest - value
                # Limit extrapolation to one nearby call, then refresh with two exact evaluations.
                if 0 < distance <= 1.25 * interval:
                    velocities = [torch.lerp(a, b, 1 + distance / interval) for a, b in zip(older, newer)]
                    self.forecasted_calls += 1
                    self.velocity_history.clear()
            if velocities is None:
                velocities = self.exact_velocities(x, value, rows)
                self.velocity_history[:] = [*self.velocity_history[-1:], (value, velocities)]
        else:
            velocities = self.exact_velocities(x, value, rows)
        predictions = {row: x - velocity * value for row, velocity in zip(rows, velocities)}
        return combine_predictions(predictions, self.guidance, value, self.cfg)

    def finish(self, release):
        """After run_end: free this run's inputs so the process can serve another run; optionally free its VRAM."""
        self.mapping.close()
        self.mapping = None
        self.call("conditioning_free", {"conditionings": self.conditionings})
        if release:
            self.call("release", {"keep_host_copy": True})

    def alive(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait()
            if hasattr(self, "reader"):
                self.reader.join(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
        if self.mapping is not None:
            self.mapping.close()
            self.mapping = None
        self.log.close()
