"""Offline tests for independent references, checkpoint contracts and editing."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from trace_core.references import boundary_condition, read_poses, prepare_references, validate_checkpoint
from trace_core.motion_io import write_sps, to_gravity_gauge
from trace_core.edit_requests import explicit_training_request, compile_request
from trace_core.edit_sampling import sample_edit
from trace_core.editor_events import PalmFK
from trace_core.editor import SourceEventExtractor, revise_events

ROOT = Path(__file__).resolve().parents[1]


def pose():
    value = np.zeros((2, 35), np.float32)
    value[:, 1] = 78
    value[:, 9] = value[:, 15] = 12
    return value


class ReferenceTests(unittest.TestCase):
    def test_independent_endpoints_do_not_create_velocity(self):
        refs = pose()
        refs[1, 0] = 1000
        refs[1, 5] = 90
        refs[1, 30] = 45
        reference, known = boundary_condition(refs, 80, 8, np.zeros(40), np.ones(40))
        np.testing.assert_array_equal(reference[0, :8, :2], 0)
        np.testing.assert_array_equal(reference[0, -8:, :2], 0)
        np.testing.assert_array_equal(reference[0, -8:, 3:5], np.tile([0, 1], (8, 1)))
        self.assertEqual(reference[0, -1, 35], 45)
        self.assertEqual(reference[0, 0, 35], 0)
        self.assertTrue(known[0, :8].all())
        self.assertFalse(known[0, 8:-8].any())

    def test_roundtrip_and_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            np.save(d / "default.npy", pose())
            end = pose()[0]
            end[30] = 45
            write_sps(d / "end.sps", end[None], fps=20)
            refs = prepare_references(d / "default.npy", end=d / "end.sps")
            np.testing.assert_array_equal(refs[0], pose()[0])
            np.testing.assert_array_equal(refs[1], end)
            refs = prepare_references(d / "missing.npy", start=d / "end.sps", end=d / "end.sps")
            np.testing.assert_array_equal(refs, np.repeat(end[None], 2, axis=0))
            with self.assertRaises(ValueError):
                prepare_references(d / "default.npy", end=d / "end.sps", end_frame=2)
            (d / "broken.sps").write_text("0\nroot tx:0 ty:78 tz:0\n")
            with self.assertRaises(ValueError):
                read_poses(d / "broken.sps")

    def test_incompatible_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save({"format": "sparse_events_v1"}, path)
            with self.assertRaisesRegex(ValueError, "boundary_events_v1"):
                validate_checkpoint(path, "standing")

    def test_invalid_boundary_statistics(self):
        for mean, std in ((np.zeros(39), np.ones(40)), (np.zeros(40), np.zeros(40)),
                          (np.zeros(40), np.full(40, np.inf))):
            with self.assertRaises(ValueError):
                boundary_condition(pose(), 80, 8, mean, std)
        with self.assertRaises(ValueError):
            boundary_condition(pose(), 80, 0, np.zeros(40), np.ones(40))


class EditingTests(unittest.TestCase):
    def test_mask_and_relative_request(self):
        delta, scope, request = explicit_training_request(
            "Raise the right hand by 8 cm during the middle half.", 80)
        self.assertEqual(delta["time_span"], [20, 60])
        self.assertTrue(scope[20:60, 33:40].all())
        self.assertFalse(scope[:, :33].any())
        self.assertFalse(scope[:20].any())
        self.assertFalse(request[:, 0].any())
        self.assertAlmostEqual(float(request[39, 1]), .08, delta=.001)
        with self.assertRaises(ValueError):
            compile_request(dict(delta, time_span=[0, 2]), 80)
        with self.assertRaises(ValueError):
            explicit_training_request("Raise both arms.", 80)
        for text in ("Do not raise the right hand by 8 cm during the middle half.",
                     "Raise the right hand by 8 cm during the middle half, then kick."):
            with self.assertRaises(ValueError):
                explicit_training_request(text, 80)

    def test_sampler_preserves_coordinates(self):
        class Model:
            def __call__(self, x, *args, **kw):
                return (torch.ones_like(x), None, None)
        source = np.repeat(to_gravity_gauge(pose())[:1], 104, axis=0)[None]
        delta, scope, request = explicit_training_request(
            "Raise the right hand by 8 cm during the middle half.", 80)
        result = sample_edit(Model(), source, np.array([80]), np.zeros((1, 2560), np.float32),
            scope[None], request[None], np.zeros((1, 26, 16), np.float32),
            np.zeros(40, np.float32), np.ones(40, np.float32), ["test"], steps=3)
        mask = ~scope[:80]
        np.testing.assert_array_equal(result[0, :80][mask], source[0, :80][mask])
        self.assertTrue(np.isfinite(result).all())

    def test_source_events_and_palm_fk(self):
        xml = ROOT / "assets/unitree_g1/g1_mocap_29dof.xml"
        source = np.zeros((1, 104, 40), np.float32)
        source[0, :80] = np.repeat(to_gravity_gauge(pose())[:1], 80, axis=0)
        events, confidence, palms, ground = SourceEventExtractor("cpu", xml)(source, [80])
        self.assertEqual(events.shape, (1, 26, 16))
        self.assertTrue(np.isfinite(palms).all())
        delta, scope, request = explicit_training_request(
            "Raise the right hand by 8 cm during the middle half.", 80)
        revised, _ = revise_events(source, [80], events, confidence, palms, ground, request[None])
        np.testing.assert_array_equal(revised[..., :12], events[..., :12])
        self.assertTrue(np.isfinite(revised).all())


if __name__ == "__main__":
    unittest.main()
