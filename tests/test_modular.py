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
from contextlib import nullcontext
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
        session.velocity_history = []
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
        session.call = Mock(side_effect=[{"shm": {"name": "region", "bytes": 20480}}, {"id": 7}, {"freed": True}])
        with patch.object(component, "attach_region", return_value=MagicMock()):
            self.assertEqual(session.import_conditioning(torch.zeros(1, 1, 4096)), 7)
        imported = session.call.call_args_list[1].args[1]
        self.assertEqual(imported["shm"], "region")
        self.assertEqual(imported["shape"], [1, 4096])
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

    def test_engine_references_respect_server_side_and_finiteness_limits(self):
        def cond(latent):
            return [[torch.zeros(1, 1, 4096), {"reference_latents": [latent], "reference_latents_method": sampler.REFERENCE_METHOD}]]
        valid = torch.ones(1, 128, 8, 1024)
        self.assertEqual(len(sampler.engine_references(cond(valid), cond(valid.clone()))), 1)
        for latent in (torch.ones(1, 128, 8, 1025), torch.ones(1, 128, 1025, 8), torch.ones(1, 128, 0, 8),
                       torch.full((1, 128, 2, 2), float("nan")), torch.full((1, 128, 2, 2), float("inf"))):
            with self.subTest(shape=tuple(latent.shape)):
                self.assertIsNone(sampler.engine_references(cond(latent), cond(latent)))
        self.assertIsNone(sampler.engine_references(cond(valid), cond(valid.to(torch.float16))))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_engine_references_compare_across_devices(self):
        def cond(latent):
            return [[torch.zeros(1, 1, 4096), {"reference_latents": [latent], "reference_latents_method": sampler.REFERENCE_METHOD}]]
        latent = torch.ones(1, 128, 2, 2)
        self.assertEqual(len(sampler.engine_references(cond(latent), cond(latent.cuda()))), 1)
        self.assertIsNone(sampler.engine_references(cond(latent), cond(latent.cuda() * 2)))

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
        # One kl_optimal step at 0.5 < denoise < 1 would otherwise run as a full denoise.
        with self.assertRaisesRegex(ValueError, "kl_optimal"):
            sampler.sampling_sigmas(sampling, "kl_optimal", 1, 0.6)
        partial = sampler.sampling_sigmas(sampling, "kl_optimal", 1, 0.5)
        self.assertEqual(len(partial), 2)
        self.assertLess(float(partial[0]), float(sampling.sigma_max))
        # An unbounded int(steps / denoise) schedule is rejected before allocating it.
        with patch.object(sampler.comfy.samplers, "calculate_sigmas") as calculate:
            with self.assertRaisesRegex(ValueError, "denoise"):
                sampler.sampling_sigmas(sampling, "simple", 1000, 1e-6)
            calculate.assert_not_called()
        self.assertGreaterEqual(len(sampler.sampling_sigmas(sampling, "simple", 1000, 0.01)), 2)
        with patch.object(sampler.comfy.samplers, "calculate_sigmas", return_value=torch.tensor([float("nan"), 0.0])):
            with self.assertRaisesRegex(ValueError, "sigma"):
                sampler.sampling_sigmas(sampling, "simple", 1, 1)

    def test_finish_frees_run_inputs_and_optionally_releases_vram(self):
        for release in (False, True):
            with self.subTest(release=release):
                session = component.ComponentSession.__new__(component.ComponentSession)
                session.mapping = Mock()
                mapping = session.mapping
                session.conditionings = [7, 8]
                session.call = Mock(return_value={})
                session.finish(release)
                mapping.close.assert_called_once_with()
                self.assertIsNone(session.mapping)
                expected = [(("conditioning_free", {"conditionings": [7, 8]}),)]
                expected += [(("release", {"keep_host_copy": True}),)] if release else []
                self.assertEqual(session.call.call_args_list, expected)

    def test_import_conditioning_frees_its_scratch_region(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = SimpleNamespace(pid=123)
        replies = {"shm_alloc": {"shm": {"name": "r1"}}, "conditioning_import": {"id": 5}, "shm_free": {"freed": True}}
        session.call = Mock(side_effect=lambda method, params: replies[method])
        region = type("Region", (bytearray,), {"close": lambda self: None})(4096 + 4 * 4096)
        with patch.object(component, "attach_region", return_value=region):
            self.assertEqual(session.import_conditioning(torch.zeros(1, 1, 4096)), 5)
        self.assertEqual(session.call.call_args_list[-1].args, ("shm_free", {"name": "r1"}))

    def keep_engine_runs(self, modes, fail_at=None, dit_file=None, kept=None, lokr=None, latent=None):
        sampling = Flow(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1.0}))
        model = SimpleNamespace(patches={}, model_options={}, get_model_object=lambda name: sampling if name == "model_sampling" else Flux2())
        latent = latent or {"samples": torch.zeros(1, 128, 2, 2)}
        cond = [[torch.zeros(1, 1, 4096), {}]]
        sessions = []

        def create(executable, load):
            session = Mock()
            session.call.return_value = {"report": {}}
            session.denoise.side_effect = lambda x, sigma: x * 0.5
            session.process.poll.return_value = None
            sessions.append(session)
            return session

        sampler.cached_engine = kept
        lokr = lokr or (lambda model, path, *args, **kwargs: nullcontext((dit_file or path, {})))
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ("moss-server.exe" if os.name == "nt" else "moss-server")
            executable.touch()
            with patch.multiple(sampler, model_file=Mock(return_value="base"), bundle_path=Mock(return_value="bundle"),
                                component_metadata_file=Mock(return_value="component"), executable_path=Mock(return_value=executable),
                                ComponentSession=Mock(side_effect=create)), \
                 patch.object(sampler.folder_paths, "get_temp_directory", return_value=directory), \
                 patch.object(sampler, "cached_lokr", Mock(side_effect=lokr)):
                for index, mode in enumerate(modes):
                    if index == fail_at:
                        sessions[-1].denoise.side_effect = RuntimeError("boom")
                        with self.assertRaises(RuntimeError):
                            sampler.MossImageCppSampler.execute(model, cond, cond, latent, 42, 2, 4, "euler", "simple", 1.0, keep_engine=mode)
                    else:
                        sampler.MossImageCppSampler.execute(model, cond, cond, latent, 42, 2, 4, "euler", "simple", 1.0, keep_engine=mode)
        return sessions

    def test_keep_engine_reuses_one_process_and_frees_each_run(self):
        sessions = self.keep_engine_runs(["release_vram", "release_vram", "keep_vram"])
        self.assertEqual(len(sessions), 1)
        self.assertEqual([c.args for c in sessions[0].finish.call_args_list], [(True,), (True,), (False,)])
        sessions[0].close.assert_not_called()
        self.assertIs(sampler.cached_engine[1], sessions[0])
        sampler.cached_engine = None

    def test_keep_engine_off_closes_cached_and_own_process(self):
        sessions = self.keep_engine_runs(["release_vram", "off"])
        self.assertEqual(len(sessions), 2)
        sessions[0].close.assert_called_once_with()
        sessions[1].close.assert_called_once_with()
        sessions[1].finish.assert_not_called()
        self.assertIsNone(sampler.cached_engine)

    def test_keep_engine_off_drops_a_kept_engine_before_lokr_cache_and_export(self):
        kept = Mock()
        seen = {}

        def lokr(model, path, *args, before_build=None, **kwargs):
            seen["closed"] = kept.close.called and sampler.cached_engine is None
            seen["before_build"] = before_build
            return nullcontext((path, {}))

        self.keep_engine_runs(["off"], kept=("old", kept), lokr=lokr)
        self.assertTrue(seen["closed"])
        self.assertIs(seen["before_build"], sampler.drop_cached_session)

    def test_lokr_cache_miss_drops_a_kept_engine_before_building(self):
        kept = Mock()

        def lokr(model, path, *args, before_build=None, **kwargs):
            self.assertFalse(kept.close.called)  # keep_vram keeps it until the cache knows it misses
            before_build()
            kept.close.assert_called_once_with()
            return nullcontext(("merged-new.safetensors", {}))

        sessions = self.keep_engine_runs(["keep_vram"], kept=("old", kept), lokr=lokr)
        self.assertEqual(len(sessions), 1)
        self.assertIs(sampler.cached_engine[1], sessions[0])
        sampler.cached_engine = None

    def test_half_precision_latent_is_sampled_in_float32_as_native(self):
        samples = (torch.randn(1, 128, 2, 2, generator=torch.Generator().manual_seed(3)) * 0.1).to(torch.bfloat16)
        latent = {"samples": samples}
        sessions = self.keep_engine_runs(["off"], latent=latent)
        x = sessions[0].denoise.call_args_list[0].args[0]
        sampling = Flow(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1.0}))
        sigmas = sampler.sampling_sigmas(sampling, "simple", 2, 1.0)
        noise = sampler.comfy.sample.prepare_noise(samples, 42).to(torch.float32)
        expected = sampling.noise_scaling(sigmas[0], noise, samples.to(torch.float32), bool(sigmas[0] >= sampling.sigma_max))
        self.assertEqual(x.dtype, torch.float32)
        torch.testing.assert_close(x.cpu(), expected, rtol=0, atol=0)
        self.assertEqual(latent["samples"].dtype, torch.bfloat16)
        self.assertIs(latent["samples"], samples)

    def test_dead_server_reports_stderr_and_close_releases_everything(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = Mock(pid=123)
        session.process.poll.return_value = 1
        session.process.stdin.write.side_effect = OSError(22, "Invalid argument")
        session.process.stdin.close.side_effect = OSError(22, "Invalid argument")
        session.sequence = 0
        session.messages = component.queue.Queue()
        session.messages.put(None)
        session.log = tempfile.TemporaryFile()
        session.log.write(b"CUDA out of memory")
        session.mapping = Mock()
        mapping, log = session.mapping, session.log
        with patch.object(component.mm, "throw_exception_if_processing_interrupted"):
            with self.assertRaisesRegex(RuntimeError, "exited: CUDA out of memory"):
                session.call("run_begin", {})
        session.process.stdin.flush.side_effect = BrokenPipeError
        session.process.stdin.write.side_effect = None
        session.messages.put(None)
        with patch.object(component.mm, "throw_exception_if_processing_interrupted"):
            with self.assertRaisesRegex(RuntimeError, "exited"):
                session.call("run_begin", {})
        session.close()
        session.process.stdout.close.assert_called_once_with()
        mapping.close.assert_called_once_with()
        self.assertIsNone(session.mapping)
        self.assertTrue(log.closed)

    def test_velocity_history_is_released_in_the_tail_finish_and_next_begin(self):
        sigmas = torch.linspace(1, 0, 11)
        x = torch.zeros(1, 128, 1, 1)
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = SimpleNamespace(pid=123)
        session.import_conditioning = Mock(side_effect=[7, 8, 9, 10])
        session.call = Mock(return_value={"shm": {}, "tensors": {}})
        session.exact_velocities = Mock(side_effect=lambda x, sigma, rows: [torch.full_like(x, sigma) for row in rows])
        with patch.object(component, "attach_region"):
            session.begin(None, None, 16, 16, sigmas, 2, forecast_denoise=True)
            for sigma in sigmas[:-4]:  # ends right after an exact step, so two velocities are held
                session.denoise(x, sigma[None])
            self.assertEqual(len(session.velocity_history), 2)
            session.denoise(x, sigmas[-3][None])  # a tail step
            self.assertEqual(session.velocity_history, [])
            session.velocity_history = [(0.5, [x])]
            session.finish(False)
            self.assertEqual(session.velocity_history, [])
            session.velocity_history = [(0.5, [x])]
            session.begin(None, None, 16, 16, sigmas, 2, forecast_denoise=False)
        self.assertEqual(session.velocity_history, [])
        self.assertEqual(session.forecast_rows, [])

    def test_post_slg_without_valid_layers_takes_no_engine_row(self):
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = SimpleNamespace(pid=123)
        session.import_conditioning = Mock(side_effect=[7, 8])
        session.call = Mock(return_value={"shm": {}, "tensors": {}})
        empty = {"layers": [], "sigma_start": 1.0, "sigma_end": 0.0, "scale": 2.0, "rescaling_scale": 0.5}
        guidance = {"simple": None, "post": [empty, empty | {"layers": [22]}]}
        with patch.object(component, "attach_region"):
            session.begin(None, None, 16, 16, torch.tensor([1.0, 0.0]), 4, guidance=guidance)
        rows = session.call.call_args.args[1]["rows"]
        self.assertEqual(rows, [{"conditioning": 7}, {"conditioning": 8}, {"conditioning": 7, "skip_layers": [22]}])
        self.assertEqual(component.prediction_rows(guidance, 0.5, 4), [0, 1, 2])

    def test_failed_housekeeping_keeps_the_finished_image(self):
        session = Mock()
        session.finish.side_effect = RuntimeError("release failed")
        sampler.cached_engine = None
        sampler.finish_session("key", session, "release_vram", True)
        session.close.assert_called_once_with()
        self.assertIsNone(sampler.cached_engine)

    def test_early_native_fallback_drops_a_kept_engine(self):
        cached = Mock()
        sampler.cached_engine = ("key", cached)
        embeddings = torch.zeros(1, 10, 4096)
        positive = [[embeddings, {"reference_latents": [torch.zeros(1, 128, 4, 4)]}]]
        native = ({"samples": torch.ones(1, 128, 4, 4)},)
        with patch.object(sampler.comfy_nodes, "common_ksampler", return_value=native):
            sampler.MossImageCppSampler.execute("model", positive, [[embeddings, {}]], {"samples": torch.zeros(1, 128, 4, 4)},
                                                42, 3, 4, "er_sde", "simple", 1.0)
        cached.close.assert_called_once_with()
        self.assertIsNone(sampler.cached_engine)

    def test_keep_engine_reuses_a_lokr_cache_slot(self):
        # Each LoKr combination has its own cache file, so a kept engine never sees rewritten weights.
        sessions = self.keep_engine_runs(["keep_vram", "keep_vram"], dit_file="merged-0123456789abcdef.safetensors")
        self.assertEqual(len(sessions), 1)
        sessions[0].close.assert_not_called()
        sampler.cached_engine = None

    def test_session_key_tracks_dit_file_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            dit = Path(directory) / "dit.safetensors"
            dit.write_bytes(b"a")
            first = sampler.session_key("server", {"dit_file": str(dit)})
            os.utime(dit, ns=(1, 1))
            self.assertNotEqual(sampler.session_key("server", {"dit_file": str(dit)}), first)

    def test_keep_engine_drops_process_after_a_failed_run(self):
        sessions = self.keep_engine_runs(["release_vram", "release_vram", "release_vram"], fail_at=1)
        self.assertEqual(len(sessions), 2)
        sessions[0].close.assert_called_once_with()
        self.assertIs(sampler.cached_engine[1], sessions[1])
        sampler.cached_engine = None

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

    def test_model_file_allows_linked_file_under_diffusion_models(self):
        config = SimpleNamespace(unet_config={"image_model": "moss_image"})
        model = SimpleNamespace(model=SimpleNamespace(model_config=config), object_patches={},
                                model_options={"transformer_options": {}}, weight_wrapper_patches={})
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "model.safetensors"
            target.touch()
            link_dir = Path(root) / "linked"
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(outside, str(link_dir))
            else:
                os.symlink(outside, link_dir)
            try:
                linked = link_dir / "model.safetensors"
                model.cached_patcher_init = (sampler.comfy.sd.load_diffusion_model, (str(linked), {}))
                with patch.object(sampler.folder_paths, "get_folder_paths", return_value=[root]):
                    self.assertEqual(sampler.model_file(model), os.path.abspath(str(linked)))
                    escaped = str(Path(root) / ".." / Path(outside).name / "model.safetensors")
                    model.cached_patcher_init = (sampler.comfy.sd.load_diffusion_model, (escaped, {}))
                    with self.assertRaisesRegex(ValueError, "diffusion_models"):
                        sampler.model_file(model)
            finally:
                (os.rmdir if os.name == "nt" else os.unlink)(link_dir)

    def test_component_call_extends_idle_deadline_on_events_and_times_out_when_hung(self):
        self.assertEqual(component.CALL_IDLE_TIMEOUT, 1800)
        session = component.ComponentSession.__new__(component.ComponentSession)
        session.process = SimpleNamespace(stdin=Mock())
        session.sequence = 0
        now = [0.0]
        replies = [b'{"event": "progress"}\n', b'{"id": 1, "result": {"ok": true}}\n']

        def next_message(timeout):
            now[0] += 1200.0
            if replies:
                return replies.pop(0)
            raise component.queue.Empty

        session.messages = SimpleNamespace(get=next_message)
        with patch.object(component.mm, "throw_exception_if_processing_interrupted"), \
                patch.object(component.time, "monotonic", side_effect=lambda: now[0]):
            self.assertEqual(session.call("load"), {"ok": True})
            with self.assertRaises(TimeoutError):
                session.call("load")

    def test_missing_reference_capability_is_memoized_until_executable_changes(self):
        sampling = Flow(SimpleNamespace(sampling_settings={"shift": 5.0, "multiplier": 1.0}))
        model = SimpleNamespace(patches={}, model_options={}, get_model_object=lambda name: sampling if name == "model_sampling" else Flux2())
        latent = {"samples": torch.zeros(1, 128, 2, 2)}
        ref = torch.ones(1, 128, 2, 2)
        cond = [[torch.zeros(1, 1, 4096), {"reference_latents": [ref], "reference_latents_method": sampler.REFERENCE_METHOD}]]
        native = ({"samples": torch.ones(1, 128, 2, 2)},)
        old_session = Mock(reference_latents=False)
        new_session = Mock(reference_latents=True)
        new_session.call.return_value = {"report": {}}
        new_session.denoise.side_effect = lambda x, sigma: x * 0.5
        sampler.cached_engine = None
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sampler.server_reference_support, {}, clear=True):
            executable = Path(directory) / ("moss-server.exe" if os.name == "nt" else "moss-server")
            executable.write_bytes(b"v1")
            sessions = Mock(side_effect=[old_session, new_session])
            with patch.multiple(sampler, model_file=Mock(return_value="base"), bundle_path=Mock(return_value="bundle"),
                                component_metadata_file=Mock(return_value="component"), executable_path=Mock(return_value=executable),
                                ComponentSession=sessions), \
                 patch.object(sampler.folder_paths, "get_temp_directory", return_value=directory), \
                 patch.object(sampler.mm, "unload_all_models") as unload, \
                 patch.object(sampler.comfy_nodes, "common_ksampler", return_value=native) as common:
                out1 = sampler.MossImageCppSampler.execute(model, cond, cond, latent, 42, 1, 4, "euler", "simple", 1.0)
                self.assertIs(out1.result[0], native[0])
                self.assertEqual(unload.call_count, 1)
                self.assertEqual(sessions.call_count, 1)
                old_session.close.assert_called_once_with()

                kept = Mock()
                sampler.cached_engine = ("key", kept)
                out2 = sampler.MossImageCppSampler.execute(model, cond, cond, latent, 42, 1, 4, "euler", "simple", 1.0)
                self.assertIs(out2.result[0], native[0])
                kept.close.assert_called_once_with()
                self.assertIsNone(sampler.cached_engine)
                self.assertEqual(unload.call_count, 1)
                self.assertEqual(sessions.call_count, 1)
                self.assertEqual(common.call_count, 2)

                executable.write_bytes(b"v2-updated")
                out3 = sampler.MossImageCppSampler.execute(model, cond, cond, latent, 42, 1, 4, "euler", "simple", 1.0)
                self.assertEqual(unload.call_count, 2)
                self.assertEqual(sessions.call_count, 2)
                self.assertEqual(common.call_count, 2)
                self.assertTrue(torch.isfinite(out3.result[0]["samples"]).all())

    def test_non_cuda_or_hip_falls_back_to_native_before_unloading(self):
        embeddings = torch.zeros(1, 10, 4096)
        cond = [[embeddings, {}]]
        latent = {"samples": torch.zeros(1, 128, 4, 4)}
        native = ({"samples": torch.ones(1, 128, 4, 4)},)
        for hip, cuda_available in ((None, False), ("6.1", True)):
            with self.subTest(hip=hip, cuda_available=cuda_available):
                kept = Mock()
                sampler.cached_engine = ("key", kept)
                with patch.object(sampler.torch.version, "hip", hip), \
                        patch.object(sampler.torch.cuda, "is_available", return_value=cuda_available), \
                        patch.object(sampler.mm, "unload_all_models") as unload, \
                        patch.object(sampler, "model_file") as model_file, \
                        patch.object(sampler.comfy_nodes, "common_ksampler", return_value=native) as common:
                    output = sampler.MossImageCppSampler.execute("model", cond, cond, latent, 42, 3, 4, "er_sde", "simple", 1.0)
                common.assert_called_once_with("model", 42, 3, 4, "er_sde", "simple", cond, cond, latent, denoise=1.0)
                unload.assert_not_called()
                model_file.assert_not_called()
                kept.close.assert_called_once_with()
                self.assertIsNone(sampler.cached_engine)
                self.assertIs(output.result[0], native[0])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
