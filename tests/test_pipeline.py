"""无需下载模型的单元测试：校验 Plan 编译、动作格式和后处理边界。"""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from trace_core.llm_compile import compile_response, canonical_parse
from trace_core.events import pack_events
from trace_core.motion_io import from_gravity_gauge, write_sps, parse_sps_file, convert_to_35dim
from trace_core.postprocess import smooth_pose, lowpass, acceleration_energy


class PlanTests(unittest.TestCase):
    def test_plan_and_padding(self):
        obj = dict(frames=9, events=[[1, 1, 0, 1, 1, 0, 0], [3, 0, 1, 2, 2, 3, 3]])
        p, known, status = compile_response(json.dumps(obj))
        self.assertFalse(status['failed'])
        self.assertEqual(p.shape, (128, 16))
        self.assertTrue(known[:3].all())
        np.testing.assert_array_equal(p[3:], 0)
        row = pack_events(dict(num_frames=9, condition_embeddings=torch.ones(1, 2560)), p)
        self.assertEqual(row['condition_embeddings'].shape, (9, 5120))
        self.assertEqual(row['phase_ids'].shape, (3,))
        self.assertEqual(row['condition_mask'].sum().item(), 2)

    def test_unknown(self):
        p, known, status = compile_response('{"frames":4,"events":[[1,-1,-1,-1,-1,-1,-1]]}')
        self.assertFalse(status['failed'])
        self.assertFalse(known.any())
        self.assertEqual(p[0, 0], .5)
        for a, b in ((2, 5), (5, 8), (8, 12), (12, 16)):
            self.assertAlmostEqual(float(p[0, a:b].sum()), 1.)

    def test_invalid_timeline_rejected(self):
        _, _, status = compile_response('{"frames":8,"events":[[1,1,1,1,1,0,0]]}')
        self.assertTrue(status['failed'])

    def test_merge_only(self):
        obj, merged = canonical_parse('{"frames":8,"events":[[1,1,1,1,1,0,0],[2,1,1,1,1,0,0]]}')
        self.assertTrue(merged)
        self.assertEqual(obj['events'], [[2, 1, 1, 1, 1, 0, 0]])


class MotionTests(unittest.TestCase):
    def pose(self, n=80):
        motion = np.zeros((n, 40), np.float32)
        motion[:, 2] = 80
        motion[:, 4] = 1
        motion[:, 5:11] = [1, 0, 0, 0, 1, 0]
        motion[:, 0] = 1
        return from_gravity_gauge(motion)

    def test_sps_roundtrip(self):
        pose = self.pose()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'motion.sps'
            write_sps(path, pose, fps=20)
            got = convert_to_35dim(parse_sps_file(path))
            np.testing.assert_allclose(pose, got, atol=1e-5)
        self.assertEqual(pose[-1, 2], 79)
        self.assertEqual(pose[-1, 1], 80)

    def test_smoothing(self):
        pose = self.pose()
        pose[:, 6] = 3*np.sin(np.arange(len(pose))*2.4)
        result, report = smooth_pose(pose)
        np.testing.assert_array_equal(result[[0, -1]], pose[[0, -1]])
        self.assertLess(report['joint_acceleration_after'], report['joint_acceleration_before'])
        self.assertFalse(report['ik'])

    def test_short_clips(self):
        for n in (1, 2, 3, 4, 5, 12):
            out, _ = smooth_pose(self.pose(n))
            self.assertEqual(out.shape, (n, 35))
            self.assertTrue(np.isfinite(out).all())

    def test_turn_crossing(self):
        pose = self.pose()
        yaw = np.linspace(160, 210, len(pose))
        pose[:, 5] = (yaw+180) % 360-180
        out, _ = smooth_pose(pose)
        unwrapped = np.unwrap(np.deg2rad(out[:, 5]))
        self.assertLess(float(np.max(np.abs(np.diff(unwrapped)))), .05)

    def test_bad_cutoff(self):
        with self.assertRaises(ValueError):
            lowpass(np.zeros((5, 2)), 20, 10)


if __name__ == '__main__':
    unittest.main()
