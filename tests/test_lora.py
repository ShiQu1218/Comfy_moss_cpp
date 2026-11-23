import tempfile
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modular import sampler
from moss_plugin.lora import export_loras
from comfy.weight_adapter.lora import LoRAAdapter
from comfy.weight_adapter.loha import LoHaAdapter
from comfy.weight_adapter.lokr import LoKrAdapter
from types import SimpleNamespace
from unittest.mock import patch
import comfy.lora
import comfy.utils
import torch


class LoraTests(unittest.TestCase):
    def test_two_adapters_across_178_materialized_targets_share_one_file(self):
        base = torch.randn(3, 4)
        lora = LoRAAdapter(set(), (torch.randn(3, 2), torch.randn(2, 4), None,
                                   None, torch.ones(3, 1), None))
        lokr = LoKrAdapter(set(), (torch.ones(1, 1), torch.randn(3, 4), None,
                                   None, None, None, None, None, None))
        patches = {f"diffusion_model.layers.{i}.proj.o.weight":
                   [(0.7, lora, 1, None, None), (-0.3, lokr, 1, None, None)]
                   for i in range(178)}
        model = SimpleNamespace(model=object(), backup={}, hook_backup={})
        with tempfile.TemporaryDirectory() as temporary:
            with patch("comfy.model_patcher.get_key_weight", return_value=(base, None, None)):
                specs = export_loras(patches, temporary, model=model)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0]["strength"], 1.0)
            tensors = comfy.utils.load_torch_file(specs[0]["path"], safe_load=True)
            self.assertEqual(len(tensors), 178)
            for key, stack in patches.items():
                expected = comfy.lora.calculate_weight(stack, base.clone(), key)
                torch.testing.assert_close(base + tensors[key[:-7] + ".diff"], expected)

    def test_materialization_uses_pristine_backups_and_reports_native_failure(self):
        key = "diffusion_model.layers.0.proj.o.weight"
        base = torch.randn(3, 4)
        adapter = LoRAAdapter(set(), (torch.randn(3, 2), torch.randn(2, 4), None, None, torch.ones(3, 1), None))
        stack = [(0.7, adapter, 1, None, None)]
        model = SimpleNamespace(model=object(), backup={key: SimpleNamespace(weight=base + 1)}, hook_backup={key: (base, None)})
        expected = comfy.lora.calculate_weight(stack, base.clone(), key)
        with tempfile.TemporaryDirectory() as temporary:
            with patch("comfy.model_patcher.get_key_weight", return_value=(base + 10, None, None)):
                spec = export_loras({key: stack}, temporary, model=model)[0]
                delta = comfy.utils.load_torch_file(spec["path"], safe_load=True)[key[:-7] + ".diff"]
                torch.testing.assert_close(base + delta, expected)
                broken = LoRAAdapter(set(), (torch.randn(3, 2), torch.randn(3, 4), None, None, torch.ones(3, 1), None))
                with self.assertLogs(level="ERROR"):
                    with self.assertRaises(ValueError):
                        export_loras({key: [(1, broken, 1, None, None)]}, temporary, model=model)

    def test_loha_lokr_native_scaling(self):
        a, b = torch.randn(6, 2), torch.randn(2, 4)
        cases = [LoHaAdapter(set(), (a, b, 3.0, a, b, None, None, None)),
                 LoKrAdapter(set(), (torch.ones(1, 1), None, 3.0, None, None, a, b, None, None)),
                 LoKrAdapter(set(), (torch.ones(1, 1), a @ b, 99.0, None, None, None, None, None, None))]
        key = "diffusion_model.layers.0.proj.o.weight"
        for adapter in cases:
            stack = [(-0.7, adapter, 1, None, None)]
            expected = comfy.lora.calculate_weight(stack, torch.zeros(6, 4), key)
            with tempfile.TemporaryDirectory() as temporary:
                spec = export_loras({key: stack}, temporary)[0]
                tensors = comfy.utils.load_torch_file(spec["path"], safe_load=True)
                mapped = comfy.lora.load_lora(tensors, {key[:-7]: key})
                actual = comfy.lora.calculate_weight([(spec["strength"], mapped[key], 1, None, None)], torch.zeros(6, 4), key)
            torch.testing.assert_close(actual, expected)

    def test_dora_locon_and_reshape_materialized_with_native_base(self):
        key = "diffusion_model.layers.0.proj.o.weight"
        base = torch.randn(3, 4)
        original = base.clone()
        up, down = torch.randn(3, 2), torch.randn(2, 4)
        cases = [LoRAAdapter(set(), (up, down, 2.0, None, torch.ones(3, 1), None)),
                 LoRAAdapter(set(), (up[:, :, None, None], down[:, :, None, None], 2.0, None, None, None)),
                 LoRAAdapter(set(), (up, down, None, None, None, list(base.shape))),
                 LoHaAdapter(set(), (up.T, down, 2.0, up.T, down, torch.eye(2)[:, :, None, None], torch.eye(2)[:, :, None, None], None)),
                 LoKrAdapter(set(), (torch.ones(1, 1, 1, 1), base, 2.0, None, None, None, None, None, None)),
                 LoKrAdapter(set(), (torch.ones(1, 1), None, 2.0, None, None, up.T, down, torch.eye(2)[:, :, None, None], None))]
        model = SimpleNamespace(model=object(), backup={}, hook_backup={})
        for adapter in cases:
            stack = [(0.5, LoRAAdapter(set(), (up, down, None, None, None, None)), 1, None, None),
                     (-0.3, adapter, 1, None, None)]
            expected = comfy.lora.calculate_weight(stack, base.clone(), key)
            with tempfile.TemporaryDirectory() as temporary:
                # Read only the requested parameter, not quantization metadata such as weight_scale.
                with patch("comfy.model_patcher.get_key_weight", return_value=(base, None, None)) as getter:
                    specs = export_loras({key: stack}, temporary, model=model)
                getter.assert_called_once_with(model.model, key)
                self.assertEqual(len(specs), 1)
                tensors = comfy.utils.load_torch_file(specs[0]["path"], safe_load=True)
                delta = tensors[key[:-7] + ".diff"]
                torch.testing.assert_close(base + delta, expected)
        torch.testing.assert_close(base, original)

    def test_native_stack_qkv_strength_and_alpha(self):
        generator = torch.Generator().manual_seed(123)
        up = torch.randn(3072, 2, generator=generator) * 0.01
        down = torch.randn(2, 3, generator=generator) * 0.01
        adapter = LoRAAdapter(set(), (up, down, 1.5, None, None, None))
        plain = LoRAAdapter(set(), (up, down, None, None, None, None))
        zero = LoRAAdapter(set(), (up, down, 0.0, None, None, None))
        key = "diffusion_model.layers.0.proj.qkv.weight"
        stack = [(0.7, adapter, 1.0, (0, 0, 3072), None),
                 (-0.25, adapter, 1.0, (0, 3072, 3072), None),
                 (1.2, plain, 1.0, (0, 6144, 3072), None),
                 (0.3, adapter, 1.0, (0, 0, 3072), None),
                 (1.0, zero, 1.0, (0, 0, 3072), None),
                 (0.0, adapter, 1.0, (0, 0, 3072), None)]
        expected = comfy.lora.calculate_weight(stack, torch.zeros(9216, 3), key)
        actual = torch.zeros_like(expected)
        with tempfile.TemporaryDirectory() as temporary:
            specs = export_loras({key: stack}, temporary)
            self.assertEqual(len(specs), 5)
            for spec in specs:
                tensors = comfy.utils.load_torch_file(spec["path"], safe_load=True)
                for part, start in (("q", 0), ("k", 3072), ("v", 6144)):
                    stem = "diffusion_model.layers.0.proj." + part
                    if stem + ".lora_up.weight" not in tensors:
                        continue
                    a, b = tensors[stem + ".lora_up.weight"], tensors[stem + ".lora_down.weight"]
                    actual[start:start + 3072] += spec["strength"] * (tensors[stem + ".alpha"] / b.shape[0]) * (a @ b)
        torch.testing.assert_close(actual, expected)

    def test_shared_factors_diff_and_no_input_mutation(self):
        up, down = torch.ones(3, 2), torch.ones(2, 4)
        adapter = LoRAAdapter(set(), (up, down, None, None, None, None))
        patches = {f"diffusion_model.layers.{i}.proj.o.weight": [(0.5, adapter, 1, None, None)] for i in (0, 1)}
        patches["diffusion_model.layers.0.proj.o.bias"] = [(-0.2, ("diff", (torch.ones(3),)), 1, None, None)]
        with tempfile.TemporaryDirectory() as temporary:
            specs = export_loras(patches, temporary)
            self.assertEqual(len(specs), 2)
            tensors = comfy.utils.load_torch_file(specs[0]["path"], safe_load=True)
            self.assertEqual(len(tensors), 6)
            diff = comfy.utils.load_torch_file(specs[1]["path"], safe_load=True)
            self.assertIn("diffusion_model.layers.0.proj.o.diff_b", diff)
            self.assertEqual(export_loras({}, temporary), [])
        self.assertTrue(torch.all(up == 1))
        self.assertTrue(torch.all(down == 1))
        self.assertEqual(len(patches), 3)

    def test_unsupported_patches_fail(self):
        key = "diffusion_model.layers.0.proj.o.weight"
        adapter = LoRAAdapter(set(), (torch.ones(3, 2), torch.ones(2, 4), None, None, None, None))
        invalid = [(1, adapter, 0.5, None, None), (1, adapter, 1, None, lambda x: x),
                   (float("nan"), adapter, 1, None, None), (1, adapter, 1, (0, 0, 1), None),
                   (1, ("set", (torch.ones(3, 4),)), 1, None, None),
                   (1, LoRAAdapter(set(), (torch.ones(3, 2), torch.ones(2, 4), None, None, torch.ones(3), None)), 1, None, None)]
        with tempfile.TemporaryDirectory() as temporary:
            for patch in invalid:
                with self.assertRaises(ValueError):
                    export_loras({key: [patch]}, temporary)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
