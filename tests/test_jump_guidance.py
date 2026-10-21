"""Jump targets must represent real sole clearance and respect source scope."""
import unittest
import numpy as np
import torch
from trace_core.motion_io import to_gravity_gauge
from trace_core.editor_geometry import G1SoleForwardKinematics
from trace_core.foot_labels import FootGeometry
from trace_core.jump_guidance import JumpHeightGuidance


class JumpGuidanceTests(unittest.TestCase):
    def source(self, jumping=True):
        poses = np.zeros((60, 35), np.float32)
        poses[:, 1] = 80.
        poses[:, 9] = poses[:, 15] = 12.
        raw = torch.from_numpy(to_gravity_gauge(poses))[None]
        valid = torch.ones(1, 60, dtype=torch.bool)
        feet = G1SoleForwardKinematics()(raw, valid)
        raw[..., 2] -= feet[..., 2].amin()*100
        if jumping:
            t = np.arange(60)
            raw[0, :, 2] += torch.from_numpy(np.maximum(0, 8-.5*(t-30)**2)).float()
        mask = torch.zeros_like(raw, dtype=torch.bool)
        mask[:, 12:48, 2] = True
        mask[:, 12:48, 11:23] = True
        return raw, mask

    def test_fixed_ground_target_and_early_identity(self):
        raw, mask = self.source()
        g = JumpHeightGuidance(raw, np.zeros(40), np.ones(40), mask, [12, 48], 10)
        self.assertAlmostEqual(g.target, .18, places=5)
        self.assertEqual(g.peak, 30)
        self.assertFalse(g.report["physical_rollout_verified"])
        self.assertIs(g(raw, .2), raw)

    def test_rejects_non_jump_and_invalid_gain(self):
        raw, mask = self.source(False)
        with self.assertRaisesRegex(ValueError, "detectable source jump"):
            JumpHeightGuidance(raw, np.zeros(40), np.ones(40), mask, [12, 48], 10)
        for gain in (0, -26, 26, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                JumpHeightGuidance(raw, np.zeros(40), np.ones(40), mask, [12, 48], gain)

    def test_lower_jump_uses_current_source(self):
        raw, mask = self.source()
        guidance = JumpHeightGuidance(raw, np.zeros(40), np.ones(40), mask, [12, 48], -3)
        self.assertAlmostEqual(guidance.target, .05, places=5)
        self.assertEqual(guidance.report["gain_cm"], -3)
        with self.assertRaisesRegex(ValueError, "Target sole clearance"):
            JumpHeightGuidance(raw, np.zeros(40), np.ones(40), mask, [12, 48], -8)

    def test_height_geometry_matches_renderer(self):
        poses = np.zeros((3, 35), np.float32)
        poses[:, 1] = [80, 93, 77]
        poses[:, 9] = [12, 22, 35]
        poses[:, 15] = [12, 24, 32]
        poses[:, 3:6] = [[0, 0, 0], [3, 2, 18], [1, -2, -10]]
        expected, _ = FootGeometry().sequence(poses)
        raw = torch.from_numpy(to_gravity_gauge(poses))[None]
        actual = G1SoleForwardKinematics()(raw, torch.ones(1, 3, dtype=torch.bool))
        np.testing.assert_allclose(actual[0, ..., 2].numpy(), expected[..., 2], atol=3e-6)


if __name__ == "__main__":
    unittest.main()
