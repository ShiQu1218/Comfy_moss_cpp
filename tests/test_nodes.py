import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

COMFY = Path(__import__("os").environ["COMFYUI_PATH"])
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu"]
spec = importlib.util.spec_from_file_location("moss_nodes", Path(__file__).resolve().parents[1] / "nodes.py")
nodes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nodes)


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for part in ("transformer", "text_encoder", "vae", "tokenizer"):
            folder = self.root / "test" / part
            folder.mkdir(parents=True)
            (folder / ("tokenizer.json" if part == "tokenizer" else "config.json")).write_text("{}")
        self.args = dict(bundle="test", dit="dit.safetensors", text_encoder="te.safetensors", vae="vae.safetensors",
                         prompt='蘋果 "quoted"; $(echo unsafe)', negative_prompt="", seed=42,
                         width=64, height=64, steps=1, cfg=3, shift=5, tier="high", memory_mode="min_memory")
        for target, value in [("BUNDLE_ROOT", self.root), ("executable_path", Mock(return_value=self.root / "moss-infer.exe")),
                              ("folder_paths.get_temp_directory", Mock(return_value=str(self.root))),
                              ("folder_paths.get_full_path_or_raise", Mock(side_effect=lambda kind, name: str(self.root / name))),
                              ("model_management.unload_all_models", Mock()), ("model_management.soft_empty_cache", Mock()),
                              ("model_management.throw_exception_if_processing_interrupted", Mock())]:
            patcher = patch.object(nodes, target, value) if "." not in target else patch("moss_nodes." + target, value)
            sys.modules["moss_nodes"] = nodes
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_paths_and_enums(self):
        for name in ("../outside", str(self.root.parent), "missing"):
            with self.assertRaises((ValueError, FileNotFoundError)):
                nodes.bundle_path(name)
        for key, value in (("tier", "--bad"), ("memory_mode", "bad"), ("sampler", "bad"), ("scheduler", "bad"), ("width", 63), ("height", 4096)):
            args = self.args | {key: value}
            if key == "height":
                args["width"] = 4096
            with self.assertRaises(ValueError):
                nodes.MossImageCpp.execute(**args)

    def test_image_and_argv(self):
        def launch(command, **kwargs):
            self.assertIsInstance(command, list)
            self.assertEqual(command[command.index("--prompt") + 1], self.args["prompt"])
            self.assertEqual(command[command.index("--sampler") + 1], "er_sde")
            self.assertEqual(command[command.index("--scheduler") + 1], "simple")
            Path(command[command.index("--json-report") + 1]).write_text(json.dumps({"timings_s": {"denoise": 1.0}}))
            nodes.Image.new("RGB", (64, 64), (255, 0, 128)).save(command[-1])
            return Mock(returncode=0, poll=Mock(return_value=0))
        with patch.object(nodes.subprocess, "Popen", side_effect=launch):
            image = nodes.MossImageCpp.execute(**self.args, sampler="er_sde", scheduler="simple").result[0]
        self.assertEqual(tuple(image.shape), (1, 64, 64, 3))
        self.assertEqual(image.dtype, nodes.torch.float32)
        self.assertEqual(float(image[0, 0, 0, 0]), 1.0)
        self.assertFalse(list(self.root.glob("mossimage-*")))

    def test_engine_error(self):
        def launch(command, **kwargs):
            kwargs["stderr"].write(b"model rejected")
            return Mock(returncode=1, poll=Mock(return_value=1))
        with patch.object(nodes.subprocess, "Popen", side_effect=launch):
            with self.assertRaisesRegex(RuntimeError, "model rejected"):
                nodes.MossImageCpp.execute(**self.args)

    def test_cancel_reaps_process(self):
        process = Mock(returncode=None, poll=Mock(return_value=None))
        with patch.object(nodes.subprocess, "Popen", return_value=process):
            with patch.object(nodes.model_management, "throw_exception_if_processing_interrupted", side_effect=[None, InterruptedError()]):
                with self.assertRaises(InterruptedError):
                    nodes.MossImageCpp.execute(**self.args)
        process.kill.assert_called_once()
        process.wait.assert_called_once()
        self.assertFalse(list(self.root.glob("mossimage-*")))


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
