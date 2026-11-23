import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_modular import sampler
from moss_plugin.guidance import native_guidance, prediction_rows, combine_predictions
from comfy_extras.nodes_slg import SkipLayerGuidanceDiT, SkipLayerGuidanceDiTSimple
import comfy.model_patcher
import torch


class Model:
    def __init__(self):
        self.model_options = {"transformer_options": {}}

    def clone(self):
        return copy.deepcopy(self)

    def get_model_object(self, name):
        return self

    def percent_to_sigma(self, value):
        return 1 - value

    def set_model_sampler_calc_cond_batch_function(self, callback):
        self.model_options["sampler_calc_cond_batch_function"] = callback

    def set_model_sampler_post_cfg_function(self, callback):
        comfy.model_patcher.set_model_options_post_cfg_function(self.model_options, callback)


class GuidanceTests(unittest.TestCase):
    def test_simple_window_cfg_one_and_invalid_layers(self):
        model = SkipLayerGuidanceDiTSimple.execute(Model(), 0.2, 0.8, "1", "22,22,99").result[0]
        settings = native_guidance(model.model_options)
        self.assertEqual(settings["simple"]["layers"], [22])
        self.assertEqual(prediction_rows(settings, 0.5, 4), [0, 2])
        self.assertEqual(prediction_rows(settings, 0.9, 4), [0, 1])
        self.assertEqual(prediction_rows(settings, 0.5, 1), [0])
        cond, neg, skipped = torch.randn(1, 4), torch.randn(1, 4), torch.randn(1, 4)
        torch.testing.assert_close(combine_predictions({0: cond, 2: skipped}, settings, 0.5, 4), skipped + (cond - skipped) * 4)
        torch.testing.assert_close(combine_predictions({0: cond, 1: neg}, settings, 0.9, 4), neg + (cond - neg) * 4)

    def test_full_native_formula_rescale_multiple_and_boundaries(self):
        model = SkipLayerGuidanceDiT.execute(Model(), 2, 0.2, 0.8, "", "22", 0.4).result[0]
        model = SkipLayerGuidanceDiT.execute(model, 0.5, 0.0, 1.0, "", "23", 0.2).result[0]
        settings = native_guidance(model.model_options)
        predictions = {i: torch.randn(1, 4, 3, 3) for i in range(4)}
        for sigma in (0.1, 0.2, 0.5, 0.8, 0.9):
            expected = predictions[1] + (predictions[0] - predictions[1]) * 4
            for index, callback in enumerate(model.model_options["sampler_post_cfg_function"]):
                with patch("comfy.samplers.calc_cond_batch", return_value=[predictions[index + 2]]):
                    expected = callback({"model": model, "cond_denoised": predictions[0], "cond": [],
                                         "denoised": expected, "sigma": torch.tensor([sigma], dtype=torch.float64),
                                         "input": torch.zeros_like(expected), "model_options": model.model_options})
            torch.testing.assert_close(combine_predictions(predictions, settings, sigma, 4), expected)

    def test_unknown_callbacks_rejected(self):
        for options in ({"transformer_options": {}, "sampler_post_cfg_function": [lambda x: x]},
                        {"transformer_options": {"patches": {}}}, {"sampler_pre_cfg_function": []}):
            with self.assertRaises(ValueError):
                native_guidance(options)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
