"""Check packaged references, citation, and filename-only dataset lists."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from trace_core.motion_io import to_gravity_gauge
from trace_core.references import boundary_condition, prepare_references, read_poses, resolve_idle_references
from tools.dataset_splits import read_splits

ROOT = Path(__file__).resolve().parents[1]


class PackagedExamplesTests(unittest.TestCase):
    def test_defaults_use_bundled_sps_not_downloaded_pose(self):
        start, end = resolve_idle_references()
        self.assertEqual(start, ROOT / 'examples/poses/idle_start.sps')
        self.assertEqual(end, ROOT / 'examples/poses/idle_end.sps')
        refs = prepare_references(ROOT / 'missing_legacy_pose.npy', start, end)
        np.testing.assert_array_equal(refs[0], read_poses(start)[0])
        np.testing.assert_array_equal(refs[1], read_poses(end)[0])
        custom = ROOT / 'custom.npy'
        self.assertEqual(resolve_idle_references(start=custom), (custom, end))
        self.assertEqual(resolve_idle_references(end=custom), (start, custom))

    def test_idle_references_and_boundary_condition(self):
        start = ROOT / "examples/poses/idle_start.sps"
        end = ROOT / "examples/poses/idle_end.sps"
        expected_hash = "e9056a3587298ed7cf5c6dd1cffc5dd8d5afda75f335eea69d0832b2fcab0b02"
        for path in (start, end):
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected_hash)
            self.assertEqual(read_poses(path).shape, (1, 35))
        poses = prepare_references(ROOT / "unused_default.npy", start, end)
        np.testing.assert_array_equal(poses[0], poses[1])
        reference, known = boundary_condition(poses, 80, 8, np.zeros(40), np.ones(40))
        expected = np.repeat(to_gravity_gauge(poses)[:1], 8, axis=0)
        np.testing.assert_allclose(reference[0, :8], expected, atol=1e-6)
        np.testing.assert_allclose(reference[0, -8:], expected, atol=1e-6)
        self.assertTrue(known[0, :8].all())
        self.assertTrue(known[0, -8:].all())
        self.assertFalse(known[0, 8:-8].any())

    def test_idle_export_uses_bundled_pose_without_model_download(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "idle"
            subprocess.run([sys.executable, str(ROOT / "tools/create_idle_examples.py"),
                            "--output", str(output)], check=True, capture_output=True)
            for name in ("idle_start", "idle_end"):
                expected = read_poses(ROOT / f"examples/poses/{name}.sps")
                np.testing.assert_array_equal(read_poses(output / f"{name}.sps"), expected)
                np.testing.assert_array_equal(np.load(output / f"{name}.npy", allow_pickle=False), expected)

    def test_published_split_integrity(self):
        folder = ROOT / "splits/bones_seed"
        manifest = json.loads((folder / "manifest.json").read_text())
        lists = read_splits(folder)
        self.assertEqual({k: len(v) for k, v in lists.items()}, manifest["counts"])
        for name in ("train", "val", "test"):
            self.assertEqual(hashlib.sha256((folder / f"{name}.jsonl").read_bytes()).hexdigest(),
                             manifest["sha256"][name])
        self.assertEqual(sum(map(len, lists.values())), 110882)

    def test_citation_matches_readme(self):
        readme = (ROOT / "README.md").read_text()
        block = re.search(r"```bibtex\n(.*?)\n```", readme, re.S)
        self.assertIsNotNone(block)
        self.assertEqual(block.group(1).strip(), (ROOT / "CITATION.bib").read_text().strip())
        self.assertRegex(readme, r'(?:\[arXiv\]\(https://arxiv\.org/abs/2610\.21515\)|<a href="https://arxiv\.org/abs/2610\.21515">)')


if __name__ == "__main__":
    unittest.main()
