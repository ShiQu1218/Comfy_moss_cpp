"""Compare GPU INT8 merging byte-for-byte with the original CPU executable."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]

def run(helper, source, output, loras, cpu_only=False, fast_rotation=False):
    request = output.with_suffix('.json')
    request.write_text(json.dumps(dict(source=str(source), output=str(output), loras=loras)))
    started = time.perf_counter()
    env = os.environ.copy()
    env.pop('MOSS_LOKR_CPU', None)
    env.pop('MOSS_LOKR_FAST_ROTATION', None)
    if cpu_only:
        env['MOSS_LOKR_CPU'] = '1'
    if fast_rotation:
        env['MOSS_LOKR_FAST_ROTATION'] = '1'
    result = subprocess.run([str(helper), str(request)], capture_output=True, check=True, env=env)
    assert b'fallback' not in result.stderr, result.stderr
    if 'build-gpu' in str(helper) and not cpu_only:
        assert b'GPU merged' in result.stderr, 'GPU path did not execute'
    assert (b'(fast rotation)' in result.stderr) == fast_rotation, result.stderr
    return time.perf_counter() - started

def main():
    torch.manual_seed(71)
    gpu = ROOT / 'native/build-gpu/moss-lokr-cache.exe'
    cpu = gpu
    with tempfile.TemporaryDirectory(prefix='moss-gpu-test-') as tmp:
        root = Path(tmp)
        for group in (4, 16, 64, 256, 1024):
            source = root / 'base.safetensors'
            save_file({'layers.0.proj.q.weight': torch.randint(-127, 128, (3072,3072), dtype=torch.int8),
                       'layers.0.proj.q.weight_scale': torch.rand(3072) * .01 + .0001,
                       'layers.0.proj.q.comfy_quant': torch.tensor(list(json.dumps(dict(format='int8_tensorwise',convrot=True,convrot_groupsize=group)).encode()),dtype=torch.uint8)}, str(source))
            specs=[]
            for i, strength in enumerate((.7, -.3)):
                path=root/f'lora{i}.safetensors'
                save_file({'layers.0.proj.q.lokr_w1':torch.randn(32,32),
                           'layers.0.proj.q.lokr_w2_a':torch.randn(96,3),
                           'layers.0.proj.q.lokr_w2_b':torch.randn(3,96),
                           'layers.0.proj.q.alpha':torch.tensor(3.)},str(path))
                specs.append(dict(path=str(path),strength=strength))
            path = root / 'materialized.safetensors'
            save_file({'layers.0.proj.q.diff': torch.randn(3072, 3072) * .01}, str(path))
            specs.append(dict(path=str(path), strength=1.))
            a,b=root/f'cpu{group}.safetensors',root/f'gpu{group}.safetensors'
            ct=run(cpu,source,a,specs,cpu_only=True); gt=run(gpu,source,b,specs)
            assert a.read_bytes()==b.read_bytes(), f'INT8 mismatch group={group}'
            fallback=root/f'fallback{group}.safetensors'
            run(gpu,source,fallback,specs,cpu_only=True)
            assert fallback.read_bytes()==a.read_bytes(), 'CPU fallback mismatch'
            print(dict(group=group,cpu_s=ct,gpu_s=gt,identical=True),flush=True)
            if group == 256:
                fast = root / 'fast.safetensors'
                ft = run(gpu, source, fast, specs, fast_rotation=True)
                reference, candidate = (load_file(str(path)) for path in (b, fast))
                prefix = 'layers.0.proj.q.'
                rq, cq = (data[prefix + 'weight'].to(torch.int16) for data in (reference, candidate))
                rs, cs = (data[prefix + 'weight_scale'] for data in (reference, candidate))
                torch.testing.assert_close(cs, rs, atol=0, rtol=2e-5)
                assert (rq - cq).abs().max() <= 1
                print(dict(group=group, fast_s=ft, int8_changed=int((rq != cq).sum()),
                           scales_changed=int((rs != cs).sum())), flush=True)
if __name__=='__main__': main()

