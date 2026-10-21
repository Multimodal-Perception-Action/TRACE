"""Contracts for explicitly exploratory generator-based editing."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tools.experimental_reference_edit import make_mask, resample


class ReferenceResamplingTests(unittest.TestCase):
    def test_mask_protects_idle_and_global_travel(self):
        for body in ("lower_body", "right_leg", "left_leg", "torso", "whole_body"):
            mask = make_mask(80, body, 8)
            self.assertFalse(mask[:8].any())
            self.assertFalse(mask[-8:].any())
            self.assertFalse(mask[:, [0, 1, 3, 4]].any())
            self.assertTrue(mask.any())
        self.assertFalse(make_mask(80, "right_leg", 8)[:, 11:17].any())
        self.assertFalse(make_mask(80, "left_leg", 8)[:, 17:23].any())
        with self.assertRaises(ValueError):
            make_mask(16, "whole_body", 8)

    def test_reference_projection_preserves_known_values(self):
        class Model:
            def __call__(self, x, *args, **kwargs):
                return torch.ones_like(x)*.25
        source = torch.randn(1, 80, 40)
        editable = torch.from_numpy(make_mask(80, "lower_body", 8))[None]
        diff = SimpleNamespace(steps=100, alpha_bar=torch.linspace(.999, .001, 100),
                               config=SimpleNamespace(x0_clip=6))
        batch = {name: None for name in ("condition_embeddings", "condition_mask",
                 "condition_spans", "phase_ids", "frame_mask")}
        result = resample(Model(), diff, batch, source, editable, .45, 10, 2.5, 42)
        self.assertTrue(torch.equal(result[~editable], source[~editable]))
        self.assertTrue(torch.isfinite(result).all())
        np.testing.assert_allclose(result[editable].numpy(), .25)

    def test_guidance_cannot_overwrite_protected_values(self):
        class Model:
            def __call__(self, x, *args, **kwargs):
                return torch.zeros_like(x)
        calls = []
        def guidance(clean, progress, final=False):
            calls.append((progress, final))
            return torch.ones_like(clean)*.75
        source = torch.randn(1, 80, 40)
        mask = torch.from_numpy(make_mask(80, "lower_body", 8))[None]
        diff = SimpleNamespace(steps=100, alpha_bar=torch.linspace(.999, .001, 100),
                               config=SimpleNamespace(x0_clip=6))
        batch = {name: None for name in ("condition_embeddings", "condition_mask",
                 "condition_spans", "phase_ids", "frame_mask")}
        out = resample(Model(), diff, batch, source, mask, .45, 4, 2.5, 42, guidance)
        self.assertTrue(torch.equal(out[~mask], source[~mask]))
        self.assertTrue(torch.all(out[mask] == .75))
        self.assertEqual(calls[-1], (1., True))


if __name__ == "__main__":
    unittest.main()
