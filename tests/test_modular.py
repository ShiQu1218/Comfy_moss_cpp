import importlib.util
import mmap
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
import uuid
import numpy as np
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.environ["COMFYUI_PATH"])
sys.argv = [sys.argv[0], "--cpu"]
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("moss_plugin", root / "__init__.py", submodule_search_locations=[str(root)])
plugin = importlib.util.module_from_spec(spec)
sys.modules["moss_plugin"] = plugin
spec.loader.exec_module(plugin)
from moss_plugin import sampler, component
import torch
from comfy import model_sampling
from comfy.latent_formats import Flux2
from nodes import EmptyLatentImage
from comfy_extras.nodes_sd3 import EmptySD3LatentImage
from comfy_extras.nodes_flux import EmptyFlux2LatentImage


class Flow(model_sampling.ModelSamplingDiscreteFlow, model_sampling.CONST):
    pass


class ModularTests(unittest.TestCase):
    def test_official_empty_latents_keep_pixel_size_and_input(self):
        model = SimpleNamespace(get_model_object=lambda name: Flux2())
        sources = [EmptyLatentImage().generate(512, 768)[0],
                   EmptySD3LatentImage.execute(512, 768).result[0],
                   EmptyFlux2LatentImage.execute(512, 768).result[0]]
        for source in sources:
            original_shape = tuple(source["samples"].shape)
            original_keys = set(source)
            with patch.object(sampler, "model_file", return_value="unused"), patch.object(sampler, "plain_conditioning", return_value=None):
                output = sampler.MossImageCppSampler.execute(model, [], [], source, 42, 1, 4, "er_sde", "simple", 0).result[0]
            self.assertEqual(tuple(output["samples"].shape), (1, 128, 48, 32))
            self.assertEqual(set(output), {"samples"})
            self.assertFalse(torch.count_nonzero(output["samples"]))
            self.assertEqual(tuple(source["samples"].shape), original_shape)
            self.assertEqual(set(source), original_keys)

    def test_incompatible_nonempty_latent_is_not_reinterpreted(self):
        model = SimpleNamespace(get_model_object=lambda name: Flux2())
        source = {"samples": torch.ones(1, 4, 16, 16), "downscale_ratio_spacial": 8}
        with patch.object(sampler, "model_file", return_value="unused"), patch.object(sampler, "plain_conditioning", return_value=None):
            with self.assertRaisesRegex(ValueError, "LATENT"):
                sampler.MossImageCppSampler.execute(model, [], [], source, 42, 1, 4, "er_sde", "simple", 0)
        self.assertEqual(tuple(source["samples"].shape), (1, 4, 16, 16))

    def test_component_uses_skipped_rows_only_in_window(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.mapping = bytearray(6144)
        session.offsets = {"x": 4096, "out": [4608, 5120, 5632]}
        session.cfg = 4
        session.guidance = {"simple": {"layers": [22], "sigma_start": 0.8, "sigma_end": 0.2}, "post": []}
        session.forecast_denoise = False
        session.denoise_calls = 0
        for index, offset in enumerate(session.offsets["out"]):
            session.mapping[offset:offset + 512] = np.full(128, index + 1, dtype=np.float32).tobytes()
        session.call = Mock(side_effect=lambda method, params: {"rows": params["rows"]})
        x = torch.ones(1, 128, 1, 1)
        for sigma, negative, mask in ((0.5, 3, 5), (0.9, 2, 3)):
            value = float(torch.tensor(sigma))
            cond, uncond = x - value, x - value * negative
            torch.testing.assert_close(session.denoise(x, torch.tensor([sigma])), uncond + (cond - uncond) * 4)
            self.assertEqual(session.call.call_args.args[1]["rows"], mask)
        session.call = Mock(return_value={"rows": 0})
        with self.assertRaises(RuntimeError):
            session.denoise(x, torch.tensor([0.5]))

    def test_shared_memory_copies_preserve_bits_and_output_ownership(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.mapping = bytearray(8192)
        session.offsets = {"x": 4096, "out": [4608]}
        # Signed zero, a subnormal and a NaN payload expose accidental numeric conversion.
        bits = torch.tensor([-2147483648, 1, 0x7fc00001], dtype=torch.int32)
        x = bits.view(torch.float32)
        payload = bits.numpy().tobytes()

        def evaluate(method, params):
            self.assertEqual(session.mapping[4096:4108], payload)
            session.mapping[4608:4620] = payload
            return {"rows": 1}

        session.call = evaluate
        velocity, = session.exact_velocities(x, 1.0, [0])
        session.mapping[4608:4620] = bytes(12)
        self.assertTrue(torch.equal(velocity.view(torch.int32), bits))

    def test_interrupt_reaps_component_process(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = Mock(poll=Mock(return_value=None))
        session.sequence = 0
        session.log = Mock()
        session.reader = Mock()
        session.mapping = Mock()
        mapping = session.mapping
        with patch.object(component.mm, "throw_exception_if_processing_interrupted", side_effect=InterruptedError):
            try:
                with self.assertRaises(InterruptedError):
                    session.call("denoise_step", {"sigma": 1.0})
            finally:
                session.close()
        session.process.kill.assert_called_once()
        session.process.wait.assert_called_once()
        mapping.close.assert_called_once()
        session.log.close.assert_called_once()

    def test_component_request_contract(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = SimpleNamespace(pid=123)
        session.call = Mock(side_effect=[{"shm": {"name": "region", "bytes": 20480}}, {"id": 7}])
        with patch.object(component, "attach_region", return_value=MagicMock()):
            self.assertEqual(session.import_conditioning(torch.zeros(1, 1, 4096)), 7)
        self.assertEqual(session.call.call_args.args[1]["shm"], "region")
        self.assertEqual(session.call.call_args.args[1]["shape"], [1, 4096])
        session.import_conditioning = Mock(side_effect=[7, 8])
        session.call = Mock(return_value={"shm": {}, "tensors": {}})
        with patch.object(component, "attach_region", return_value=MagicMock()):
            session.begin(None, None, 1024, 1024, torch.tensor([1.0, 0.5, 0.0]), 4,
                          loras=[{"path": "adapter.safetensors", "strength": -0.5}])
        params = session.call.call_args.args[1]
        self.assertFalse(params["exact_sigma"])
        self.assertEqual(params["rows"], [{"conditioning": 7}, {"conditioning": 8}])
        self.assertEqual(params["loras"], [{"path": "adapter.safetensors", "strength": -0.5}])

    def test_denoise_forecast_is_optional_and_refreshes_when_guidance_changes(self):
        sigmas = torch.linspace(1, 0, 11)
        guidance = {"simple": {"layers": [22], "sigma_start": 0.35, "sigma_end": 0.0}, "post": []}
        x = torch.zeros(1, 128, 1, 1)
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                session = component.ComponentSession.__new__(component.ComponentSession)
                session.process = SimpleNamespace(pid=123)
                session.import_conditioning = Mock(side_effect=[7, 8])
                session.call = Mock(return_value={"shm": {}, "tensors": {}})
                session.exact_velocities = Mock(side_effect=lambda x, sigma, rows: [torch.full_like(x, 2 * sigma + row) for row in rows])
                with patch.object(component, "attach_region"):
                    session.begin(None, None, 16, 16, sigmas, 2, guidance=guidance, forecast_denoise=enabled)
                for sigma in sigmas[:-1]:
                    value = float(sigma)
                    negative = 1 if value > 0.35 else 2
                    torch.testing.assert_close(session.denoise(x, sigma[None]), torch.full_like(x, -value * (2 * value - negative)))
                exact_sigmas = [call.args[1] for call in session.exact_velocities.call_args_list]
                expected = [sigma.item() for i, sigma in enumerate(sigmas[:-1]) if not (enabled and i == 4)]
                self.assertEqual(exact_sigmas, expected)
                self.assertEqual(session.forecasted_calls, int(enabled))

    def test_reference_latents_fall_back_to_native_sampler(self):
        embeddings = torch.zeros(1, 10, 4096)
        positive = [[embeddings, {"reference_latents": [torch.zeros(1, 128, 4, 4)]}]]
        negative = [[embeddings, {}]]
        latent = {"samples": torch.zeros(1, 128, 4, 4)}
        native = ({"samples": torch.ones(1, 128, 4, 4)},)
        with patch.object(sampler.comfy_nodes, "common_ksampler", return_value=native) as common, \
                patch.object(sampler, "model_file") as model_file:
            output = sampler.MossImageCppSampler.execute("model", positive, negative, latent, 42, 3, 4, "er_sde", "simple", 0.6)
        common.assert_called_once_with("model", 42, 3, 4, "er_sde", "simple", positive, negative, latent, denoise=0.6)
        model_file.assert_not_called()
        self.assertIs(output.result[0], native[0])

    def test_engine_references_only_for_the_engine_layout(self):
        embeddings = torch.zeros(1, 10, 4096)
        reference = torch.ones(1, 128, 4, 6)
        method = sampler.REFERENCE_METHOD

        def conditioning(**metadata):
            return [[embeddings, metadata]]

        self.assertEqual(sampler.engine_references(conditioning(), conditioning()), [])
        both = conditioning(reference_latents=[reference], reference_latents_method=method)
        copy = conditioning(reference_latents=[reference.clone()], reference_latents_method=method)
        self.assertEqual(sampler.engine_references(both, copy), [reference])
        self.assertIs(sampler.plain_conditioning(both), embeddings)
        for positive, negative in [
                (both, conditioning()),  # the uncond row would attend to references it lacks in ComfyUI
                (conditioning(reference_latents=[reference]), conditioning(reference_latents=[reference])),
                (both, conditioning(reference_latents=[torch.zeros(1, 128, 4, 6)], reference_latents_method=method)),
                (conditioning(reference_latents=[reference] * 5, reference_latents_method=method),
                 conditioning(reference_latents=[reference] * 5, reference_latents_method=method)),
                (conditioning(reference_latents=[torch.ones(1, 128, 128, 129)], reference_latents_method=method),
                 conditioning(reference_latents=[torch.ones(1, 128, 128, 129)], reference_latents_method=method))]:
            self.assertIsNone(sampler.engine_references(positive, negative))

    def test_reference_runs_split_rows_into_pairs(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        x = torch.zeros(1, 128, 2, 2)
        nbytes = x.numel() * 4
        session.mapping = bytearray(4096 + 4 * nbytes)
        session.offsets = {"x": 4096, "out": [4096 + (k + 1) * nbytes for k in range(3)]}
        session.rows_per_call = 2
        masks = []

        def call(method, params):
            masks.append(params["rows"])
            return {"rows": params["rows"]}

        session.call = call
        velocities = session.exact_velocities(x, 0.5, [0, 1, 2])
        self.assertEqual(masks, [0b011, 0b100])
        self.assertEqual(len(velocities), 3)

    def test_plain_native_conditioning(self):
        embeddings = torch.zeros(1, 10, 4096)
        conditioning = [[embeddings, {"pooled_output": None, "attention_mask": torch.ones(1, 10)}]]
        self.assertIs(sampler.plain_conditioning(conditioning), embeddings)
        for value in ([[embeddings, {"control": object()}]], conditioning * 2,
                      [[embeddings, {"attention_mask": torch.zeros(1, 10)}]], [[torch.zeros(2, 10, 4096), {}]]):
            with self.assertRaises(ValueError):
                sampler.plain_conditioning(value)

    def test_reject_patches_instead_of_ignoring(self):
        config = SimpleNamespace(unet_config={"image_model": "moss_image"})
        model = SimpleNamespace(model=SimpleNamespace(model_config=config), patches={"lora": 1},
                                model_options={"model_function_wrapper": object()}, object_patches={})
        with self.assertRaisesRegex(ValueError, "patch"):
            sampler.model_file(model)
        model.model_options = {"transformer_options": {}}
        model.object_patches = {"diffusion_model": object()}
        with self.assertRaisesRegex(ValueError, "patch"):
            sampler.model_file(model)

    def test_reject_weight_wrapper_and_loader_dtype(self):
        config = SimpleNamespace(unet_config={"image_model": "moss_image"})
        model = SimpleNamespace(model=SimpleNamespace(model_config=config), object_patches={},
                                model_options={"transformer_options": {}}, weight_wrapper_patches={"weight": [object()]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.touch()
            options = {}
            model.cached_patcher_init = (sampler.comfy.sd.load_diffusion_model, (str(path), options))
            with patch.object(sampler.folder_paths, "get_folder_paths", return_value=[directory]):
                with self.assertRaisesRegex(ValueError, "weight wrapper"):
                    sampler.model_file(model)
                model.weight_wrapper_patches.clear()
                self.assertEqual(sampler.model_file(model), str(path))
                options["dtype"] = torch.float8_e4m3fn
                with self.assertRaisesRegex(ValueError, "weight_dtype"):
                    sampler.model_file(model)

    def test_sampling_schedule_sanitizes_degenerate_sigmas(self):
        sampling = Flow(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1.0}))
        sigmas = sampler.sampling_sigmas(sampling, "simple", 50, 0.01)
        self.assertEqual(len(sigmas), 11)
        self.assertTrue((sigmas[1:] < sigmas[:-1]).all())
        self.assertEqual(sampler.sampling_sigmas(sampling, "kl_optimal", 1, 1).tolist(), [1.0, 0.0])
        with patch.object(sampler.comfy.samplers, "calculate_sigmas", return_value=torch.tensor([float("nan"), 0.0])):
            with self.assertRaisesRegex(ValueError, "sigma"):
                sampler.sampling_sigmas(sampling, "simple", 1, 1)

    def test_sampler_runs_native_solvers_and_rejects_nonfinite_output(self):
        sampling = Flow(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1.0}))
        model = SimpleNamespace(patches={}, model_options={}, get_model_object=lambda name: sampling if name == "model_sampling" else Flux2())
        latent = {"samples": torch.zeros(1, 128, 2, 2)}
        cond = [[torch.zeros(1, 1, 4096), {}]]
        session = Mock()
        session.call.return_value = {"report": {}}
        session.denoise.side_effect = lambda x, sigma: x * 0.5
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ("moss-server.exe" if os.name == "nt" else "moss-server")
            executable.touch()
            with patch.multiple(sampler, model_file=Mock(return_value="base"), bundle_path=Mock(return_value="bundle"),
                                component_metadata_file=Mock(return_value="component"), executable_path=Mock(return_value=executable),
                                ComponentSession=Mock(return_value=session)), \
                 patch.object(sampler.folder_paths, "get_temp_directory", return_value=directory):
                execute = sampler.MossImageCppSampler.execute
                for steps, schedule, denoise in ((50, "simple", 0.01), (1, "kl_optimal", 1.0)):
                    for solver in sampler.CPP_SAMPLERS:
                        with self.subTest(solver=solver, schedule=schedule):
                            output = execute(model, cond, cond, latent, 42, steps, 4, solver, schedule, denoise).result[0]
                            self.assertTrue(torch.isfinite(output["samples"]).all())
                session.denoise.side_effect = lambda x, sigma: torch.full_like(x, float("nan"))
                with self.assertRaisesRegex(RuntimeError, "NaN"):
                    execute(model, cond, cond, latent, 42, 1, 4, "euler", "kl_optimal", 1.0)
                self.assertEqual(session.close.call_count, 2 * len(sampler.CPP_SAMPLERS) + 1)

    @unittest.skipUnless(os.name == "nt", "Windows named mapping check")
    def test_mapping_identity(self):
        pid = os.getpid()
        token = uuid.uuid4().hex
        name = f"Local\\mossinfer-{pid}-test-{uuid.uuid4().hex}"
        size = 8192
        original = mmap.mmap(-1, size, tagname=name)
        self.addCleanup(original.close)
        original[:48] = struct.pack("<8sII16sQQ", b"MOSSSHM\0", 1, pid, bytes.fromhex(token), size, 4096)
        region = {"name": name, "token": token, "bytes": size}
        attached = component.attach_region(region, pid)
        attached[4096:4100] = b"test"
        self.assertEqual(original[4096:4100], b"test")
        attached.close()
        with self.assertRaisesRegex(ValueError, "header"):
            component.attach_region(region | {"token": "00" * 16}, pid)
        with self.assertRaisesRegex(ValueError, "size"):
            component.attach_region(region | {"bytes": 1}, pid)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
