"""Relative crouch editing must change the current height in the requested direction."""
import unittest
import numpy as np
import torch

from trace_core.crouch_guidance import CrouchDepthGuidance, relative_crouch_target, validate_crouch_report
from trace_core.motion_io import to_gravity_gauge
from trace_core.conversation import compile_intent, parse_intent, validate_explicit_distance
import json


class CrouchGuidanceTests(unittest.TestCase):
    def heights(self):
        value = np.full(60, 80., np.float32)
        value[8:52] -= 20*np.sin(np.linspace(0, np.pi, 44))**2
        return value

    def intent(self, direction="deeper", amount=None):
        return dict(action="crouch_depth", target_caption="A person crouches more deeply.",
                    body="lower_body", direction=direction, magnitude="moderate",
                    amount_cm=amount, time_region="full", reason="Change the source crouch depth.")

    def test_signed_relative_target_and_endpoints(self):
        source = self.heights()
        for delta in [-5, 5]:
            target, weight, record = relative_crouch_target(source, [8, 52], delta)
            self.assertAlmostEqual(float(target.min()-source.min()), delta, places=5)
            np.testing.assert_array_equal(target[:8], source[:8])
            np.testing.assert_array_equal(target[-8:], source[-8:])
            self.assertEqual(weight[8], 0)
            self.assertEqual(weight[51], 0)
            self.assertEqual(record["pelvis_delta_cm"], delta)

    def test_relative_to_latest_actual_height(self):
        source = self.heights()
        first, _, _ = relative_crouch_target(source, [8, 52], -5)
        second, _, _ = relative_crouch_target(first, [8, 52], 3)
        self.assertAlmostEqual(float(second.min()-source.min()), -2, places=5)

    def test_flat_source_and_oversized_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No source crouch"):
            relative_crouch_target(np.full(60, 80), [8, 52], -5)
        for delta in [0, 16, float("nan")]:
            with self.assertRaises(ValueError):
                relative_crouch_target(self.heights(), [8, 52], delta)
        with self.assertRaisesRegex(ValueError, "at least 3 cm"):
            relative_crouch_target(80-(80-self.heights())*.5, [8, 52], 8)

    def test_parser_and_compiler_signs(self):
        summary = dict(frames=60, boundary_frames=8, measured_root_heights_cm=self.heights().tolist())
        for direction, delta in [("deeper", -5), ("shallower", 5)]:
            value = parse_intent(json.dumps(self.intent(direction)))
            result = compile_intent(value, summary)
            self.assertEqual(result["pelvis_delta_cm"], delta)
            self.assertAlmostEqual(result["target_root_minimum_cm"], float(self.heights().min())+delta)
        validate_explicit_distance(self.intent(amount=5), "Crouch 5 cm deeper.")
        with self.assertRaises(ValueError):
            validate_explicit_distance(self.intent(), "Crouch 5 cm deeper.")
        with self.assertRaises(ValueError):
            parse_intent(json.dumps(self.intent("higher")))

    def test_feasibility_guard(self):
        validate_crouch_report(dict(root_profile_max_error_cm=0, sole_reference_rmse_cm=.2))
        for root, foot in [(.2, .2), (0, 3), (0, float("nan"))]:
            with self.assertRaises(ValueError):
                validate_crouch_report(dict(root_profile_max_error_cm=root, sole_reference_rmse_cm=foot))

    def test_sampler_projection_and_protected_coordinates(self):
        torch.set_num_threads(2)
        poses = np.zeros((60, 35), np.float32)
        poses[:, 1] = self.heights()
        poses[:, 9] = poses[:, 15] = 30
        raw = torch.from_numpy(to_gravity_gauge(poses))[None]
        mean = np.linspace(-1, 1, 40).astype(np.float32)
        std = np.full(40, 10., np.float32)
        source = (raw-torch.from_numpy(mean))/torch.from_numpy(std)
        mask = torch.zeros_like(source, dtype=torch.bool)
        mask[:, 8:52, 2] = True
        mask[:, 8:52, 11:23] = True
        for delta in [-5, 5]:
            guide = CrouchDepthGuidance(source, mean, std, mask, [8, 52], delta, final_iterations=2)
            self.assertIs(guide(source, .2), source)
            with torch.inference_mode():
                output = guide(source, 1., final=True)
            self.assertTrue(torch.equal(output[~mask], source[~mask]))
            height = (output*torch.from_numpy(std)+torch.from_numpy(mean))[0, :, 2]
            self.assertAlmostEqual(float(height.min()-raw[0, :, 2].min()), delta, places=4)
            self.assertLess(guide.report["root_profile_max_error_cm"], 1e-4)


if __name__ == "__main__":
    unittest.main()
