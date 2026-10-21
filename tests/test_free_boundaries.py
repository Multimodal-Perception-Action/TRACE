"""Checks for disabling endpoint conditions without swapping generator weights."""
from pathlib import Path
import tempfile
import unittest

import torch

from trace_core.boundary import sample_with_boundaries
from trace_core.diffusion import ContinuousDiffusionConfig, SMDGaussianDiffusion
from trace_core.references import validate_checkpoint


class FreeBoundaryTests(unittest.TestCase):
    def test_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save(dict(
                format="boundary_events_v1",
                model_config=dict(feature_dim=40, condition_embedding_dim=2560,
                                  patch_size=4, max_patches=128),
                ema={},
            ), path)
            self.assertEqual(validate_checkpoint(path, "free")["format"], "boundary_events_v1")
            with self.assertRaisesRegex(ValueError, "sparse_events_v1"):
                validate_checkpoint(path, "none")
            with self.assertRaisesRegex(ValueError, "Unknown boundary"):
                validate_checkpoint(path, "invalid")

    def test_zero_mask_ignores_reference_everywhere(self):
        class Model:
            def __call__(self, x, *args, boundary_reference, boundary_mask, **kwargs):
                assert not boundary_mask.any()
                assert not boundary_reference.any()
                return torch.full_like(x, .125)

        batch = dict(condition_embeddings=torch.zeros(1, 1, 4),
                     condition_mask=torch.ones(1, 1, dtype=torch.bool),
                     condition_spans=torch.zeros(1, 1, 2),
                     phase_ids=torch.zeros(1, 3, dtype=torch.long),
                     frame_mask=torch.ones(1, 12, dtype=torch.bool))
        diff = SMDGaussianDiffusion(ContinuousDiffusionConfig(steps=20))
        known = torch.zeros(1, 12, 40, dtype=torch.bool)
        results = []
        for value in (0., 50.):
            result = sample_with_boundaries(
                Model(), diff, batch, torch.full((1, 12, 40), value), known,
                steps=4, cfg=2.5, precision="fp32",
                generator=torch.Generator().manual_seed(42),
            )
            results.append(result)
            torch.testing.assert_close(result, torch.full_like(result, .125))
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
