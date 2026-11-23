import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modular import sampler
from moss_plugin import lokr_cache as cache
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.lokr import LoKrAdapter
from filelock import FileLock, Timeout
from safetensors.torch import load_file, save_file
import torch


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.safetensors"
        self.source.write_bytes(b"base")
        self.helper = self.root / "helper.exe"
        self.helper.write_bytes(b"helper")
        self.key = "diffusion_model.layers.0.proj.o.weight"
        self.adapter = LoKrAdapter(set(), (torch.ones(1, 1), torch.ones(3, 4), None, None, None, None, None, None, None))
        self.model = SimpleNamespace(patches={self.key: [(0.7, self.adapter, 1, None, None)]})
        self.addCleanup(patch.stopall)
        patch.object(cache, "CACHE_ROOT", self.root / "cache").start()
        patch.object(cache, "HELPER", self.helper).start()
        patch.object(cache.mm, "throw_exception_if_processing_interrupted").start()
        self.builder = patch.object(cache, "build_cache_model", side_effect=lambda helper, source, output, specs: output.write_bytes(b"bake")).start()

    def load(self, enabled=True):
        return cache.cached_lokr(self.model, str(self.source), self.root, enabled=enabled)

    def test_reuse_invalidates_for_strength_adapter_source_and_helper(self):
        original = self.model.patches
        for expected in (1, 1):
            with self.load() as (path, patches):
                self.assertEqual(Path(path).read_bytes(), b"bake")
                self.assertEqual(patches, {})
            self.assertEqual(self.builder.call_count, expected)
            self.assertIs(self.model.patches, original)
        changes = [lambda: self.model.patches.update({self.key: [(0.8, self.adapter, 1, None, None)]}),
                   lambda: self.adapter.weights[1].add_(1),
                   lambda: os.utime(self.source, ns=(self.source.stat().st_atime_ns, self.source.stat().st_mtime_ns + 100000)),
                   lambda: self.helper.write_bytes(b"updated")]
        for expected, change in enumerate(changes, 2):
            change()
            with self.load():
                pass
            self.assertEqual(self.builder.call_count, expected)

    def test_fast_rotation_changes_cache_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.load():
                pass
            with patch.dict(os.environ, {"MOSS_LOKR_FAST_ROTATION": "1"}):
                for _ in range(2):
                    with self.load():
                        pass
                self.assertEqual(self.builder.call_count, 2)
            with self.load():
                pass
        self.assertEqual(self.builder.call_count, 3)

    def test_keep_plain_lora_and_bypass_unsafe_or_disabled_stacks(self):
        plain = LoRAAdapter(set(), (torch.ones(3, 2), torch.ones(2, 4), None, None, None, None))
        plain_patch = (0.4, plain, 1, None, None)
        self.model.patches[self.key].append(plain_patch)
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)) as capability:
            with self.load() as (_, remaining):
                self.assertEqual(remaining, {self.key: [plain_patch]})
            capability.return_value = (7, 5)
            with self.load() as (path, remaining):
                self.assertEqual(path, str(self.source))
                self.assertIs(remaining, self.model.patches)
        with self.load(enabled=False) as (path, remaining):
            self.assertEqual(path, str(self.source))
            self.assertIs(remaining, self.model.patches)
        nonlinear = LoRAAdapter(set(), (torch.ones(3, 2), torch.ones(2, 4), None, None, torch.ones(3), None))
        self.model.patches[self.key].append((1, nonlinear, 1, None, None))
        with self.load() as (path, remaining):
            self.assertEqual(path, str(self.source))
            self.assertIs(remaining, self.model.patches)
        self.assertEqual(self.builder.call_count, 1)

    def test_invalid_metadata_or_modified_output_is_rebuilt_and_lock_is_held(self):
        with self.load() as (path, _):
            competing = FileLock(str(cache.CACHE_ROOT / "cache.lock"))
            with self.assertRaises(Timeout):
                competing.acquire(timeout=0)
        with competing.acquire(timeout=0):
            pass
        (cache.CACHE_ROOT / "manifest.json").write_text("invalid", encoding="utf-8")
        with self.load():
            pass
        os.utime(path, ns=(Path(path).stat().st_atime_ns, Path(path).stat().st_mtime_ns + 100000))
        with self.load():
            pass
        self.assertEqual(self.builder.call_count, 3)

    def test_keep_global_low_rank_with_cached_block_lokr(self):
        key = "diffusion_model.text_embedder.blocks.0.qkv.weight"
        plain = LoRAAdapter(set(), (torch.ones(3, 2), torch.ones(2, 4), None, None, None, None))
        patch_entry = (0.4, plain, 1, None, None)
        self.model.patches[key] = [patch_entry]
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)):
            for _ in range(2):
                with self.load() as (path, remaining):
                    self.assertNotEqual(path, str(self.source))
                    self.assertEqual(remaining, {key: [patch_entry]})
            updated = (0.8, plain, 1, None, None)
            self.model.patches[key] = [updated]
            with self.load() as (_, remaining):
                self.assertEqual(remaining, {key: [updated]})
            self.assertEqual(self.builder.call_count, 1)
            self.model.patches[key].append(self.model.patches[self.key][0])
            with self.load() as (path, remaining):
                self.assertEqual(path, str(self.source))
                self.assertIs(remaining, self.model.patches)

    def test_failed_build_preserves_previous_cache_and_falls_back(self):
        with self.load():
            pass
        previous = (cache.CACHE_ROOT / "manifest.json").read_bytes()
        self.model.patches[self.key][0] = (0.9, self.adapter, 1, None, None)
        self.builder.side_effect = RuntimeError("merge failed")
        with self.assertLogs(level="WARNING"):
            with self.load() as (path, patches):
                self.assertEqual(path, str(self.source))
                self.assertIs(patches, self.model.patches)
                with FileLock(str(cache.CACHE_ROOT / "cache.lock")).acquire(timeout=0):
                    pass
        self.assertEqual((cache.CACHE_ROOT / "manifest.json").read_bytes(), previous)
        self.assertEqual(list(cache.CACHE_ROOT.glob("building-*")), [])
        with FileLock(str(cache.CACHE_ROOT / "cache.lock")).acquire(timeout=0):
            pass

    def test_source_change_during_build_is_not_committed(self):
        def changed(helper, source, output, specs):
            output.write_bytes(b"bake")
            self.source.write_bytes(b"changed")
        self.builder.side_effect = changed
        with self.assertLogs(level="WARNING"):
            with self.load() as (path, _):
                self.assertEqual(path, str(self.source))
        self.assertFalse((cache.CACHE_ROOT / "merged.safetensors").exists())

    def test_cancel_propagates_and_releases_lock(self):
        from comfy.model_management import InterruptProcessingException
        self.builder.side_effect = InterruptProcessingException()
        with self.assertRaises(InterruptProcessingException):
            with self.load():
                self.fail("cancel must not fall back to another model load")
        self.assertEqual(list(cache.CACHE_ROOT.glob("building-*")), [])
        with FileLock(str(cache.CACHE_ROOT / "cache.lock")).acquire(timeout=0):
            pass

    def test_unwritable_cache_uses_regular_load(self):
        with patch.object(Path, "mkdir", side_effect=PermissionError("read only")), self.assertLogs(level="WARNING"):
            with self.load() as (path, patches):
                self.assertEqual(path, str(self.source))
                self.assertIs(patches, self.model.patches)
        self.builder.assert_not_called()

    def test_unwritable_lock_uses_regular_load(self):
        with patch.object(FileLock, "acquire", side_effect=PermissionError("read only")), self.assertLogs(level="WARNING"):
            with self.load() as (path, patches):
                self.assertEqual(path, str(self.source))
                self.assertIs(patches, self.model.patches)
        self.builder.assert_not_called()

    def test_failed_preparation_and_lock_timeout_use_regular_load(self):
        def partial_export(patches, directory, **kwargs):
            (Path(directory) / "partial.safetensors").touch()
            raise cache.SafetensorError("disk full")

        for target, error in (("export_loras", partial_export),
                              ("fingerprint", OSError("file missing"))):
            with self.subTest(target=target), patch.object(cache, target, side_effect=error), self.assertLogs(level="WARNING"):
                with self.load() as (path, patches):
                    self.assertEqual(path, str(self.source))
                    self.assertIs(patches, self.model.patches)
                    self.assertEqual(list(self.root.glob("lokr-key-*")), [])
        with patch.object(FileLock, "acquire", side_effect=Timeout("cache.lock")), \
             patch.object(cache.time, "monotonic", side_effect=[0, 301]), self.assertLogs(level="WARNING"):
            with self.load() as (path, patches):
                self.assertEqual(path, str(self.source))
                self.assertIs(patches, self.model.patches)
        self.builder.assert_not_called()


@unittest.skipUnless(cache.HELPER.is_file(), "native helper not installed")
class NativeHelperTests(unittest.TestCase):
    def test_dora_cache_preserves_order_and_hit_skips_materialization(self):
        import comfy.lora
        torch.manual_seed(19)
        key = "diffusion_model.video_embedder.weight"
        base = torch.randn(3072, 128).to(torch.bfloat16)
        plain = LoKrAdapter(set(), (torch.randn(32, 8), torch.randn(96, 16), None,
                                    None, None, None, None, None, None))
        dora = LoKrAdapter(set(), (torch.randn(32, 8), torch.randn(96, 16), None,
                                   None, None, None, None, None, torch.ones(1, 128)))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "base.safetensors"
            save_file({"video_embedder.weight": base}, str(source))
            for index, adapters in enumerate(((plain, dora), (dora, plain))):
                stack = [(0.7, adapters[0], 1, None, None), (-0.3, adapters[1], 1, None, None)]
                model = SimpleNamespace(model=object(), backup={}, hook_backup={}, patches={key: stack})
                patched = comfy.lora.calculate_weight(stack, base.float().clone(), key)
                expected = (base.float() + (patched - base.float())).to(torch.bfloat16)
                torch.testing.assert_close(expected, patched.to(torch.bfloat16))
                with patch.object(cache, "CACHE_ROOT", root / f"cache{index}"), \
                     patch("comfy.model_patcher.get_key_weight", return_value=(base, None, None)), \
                     patch.object(cache.mm, "get_torch_device", return_value=torch.device("cpu")):
                    with cache.cached_lokr(model, source, root) as (output, remaining):
                        self.assertEqual(remaining, {})
                        torch.testing.assert_close(load_file(output)["video_embedder.weight"], expected, rtol=0, atol=0)
                    with patch("moss_plugin.lora.full_delta", side_effect=AssertionError("cache hit recalculated DoRA")):
                        with cache.cached_lokr(model, source, root):
                            pass

    def test_bf16_merge_matches_native_math_preserves_source_and_rejects_overwrite(self):
        torch.manual_seed(5)
        key = "diffusion_model.video_embedder.weight"
        w1, w2 = torch.randn(32, 8), torch.randn(96, 16)
        adapter = LoKrAdapter(set(), (w1, w2, 99.0, None, None, None, None, None, None))
        base = torch.randn(3072, 128).to(torch.bfloat16)
        expected = (base.float() + 0.7 * torch.kron(w1, w2)).to(torch.bfloat16)
        with tempfile.TemporaryDirectory(prefix="moss-快取-") as temporary:
            root = Path(temporary)
            source, output = root / "source.safetensors", root / "output.safetensors"
            save_file({"video_embedder.weight": base, "untouched": torch.arange(3)}, str(source))
            original = source.read_bytes()
            specs = cache.export_loras({key: [(0.7, adapter, 1, None, None)]}, root)
            cache.build_cache_model(cache.HELPER, source, output, specs)
            actual = load_file(str(output))
            torch.testing.assert_close(actual["video_embedder.weight"], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual["untouched"], torch.arange(3))
            self.assertEqual(source.read_bytes(), original)
            merged = output.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "new file"):
                cache.build_cache_model(cache.HELPER, source, output, specs)
            self.assertEqual(output.read_bytes(), merged)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
