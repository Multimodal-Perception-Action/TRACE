"""Filename-only split export and local resolution tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("dataset_splits", Path(__file__).parents[1] / "tools/dataset_splits.py")
splits = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(splits)


class DatasetSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        for name, filenames in {"train": ["walk.sps", "walk_M.sps"], "val": ["wave.sps"], "test": ["jump.sps"]}.items():
            rows = [{"filepath": "/private/" + f, "metadata": {"filename": Path(f).stem, "actor_age_yr": 30, "caption": "private"}}
                    for f in filenames]
            (self.source / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_export_retains_only_filenames(self):
        output = self.root / "splits"
        report = splits.export_splits(self.source, output)
        self.assertEqual(report["counts"], {"train": 2, "val": 1, "test": 1})
        self.assertEqual((output / "train.jsonl").read_text(), '{"filename": "walk.sps"}\n{"filename": "walk_M.sps"}\n')
        for path in output.iterdir():
            text = path.read_text()
            self.assertNotIn("/private", text)
            self.assertNotIn("actor_age_yr", text)
            self.assertNotIn('"caption"', text)

    def test_existing_output_is_preserved(self):
        output = self.root / "splits"
        output.mkdir()
        (output / "keep").write_text("unchanged")
        with self.assertRaises(FileExistsError):
            splits.export_splits(self.source, output)
        self.assertEqual((output / "keep").read_text(), "unchanged")

    def test_duplicate_filename_fails_before_output(self):
        (self.source / "test.jsonl").write_text('{"filepath":"/different/walk.sps"}\n')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            splits.export_splits(self.source, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_mirrors_must_share_a_split(self):
        (self.source / "test.jsonl").write_text('{"filepath":"/different/wave_M.sps"}\n')
        with self.assertRaisesRegex(ValueError, "Mirror"):
            splits.export_splits(self.source, self.root / "bad")

    def test_metadata_filename_mismatch_rejected(self):
        (self.source / "test.jsonl").write_text('{"filepath":"/private/jump.sps","metadata":{"filename":"walk"}}\n')
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            splits.export_splits(self.source, self.root / "bad")

    def prepare_motion_tree(self, suffix=".sps"):
        output, motion = self.root / "splits", self.root / "motions"
        splits.export_splits(self.source, output)
        (motion / "day1").mkdir(parents=True)
        for name in ("walk", "walk_M", "wave", "jump"):
            (motion / "day1" / (name + suffix)).write_text("synthetic fixture")
        return output, motion

    def test_resolver_keeps_relative_paths_and_order(self):
        source, motion = self.prepare_motion_tree()
        output = self.root / "resolved"
        splits.resolve_splits(source, motion, output)
        row = json.loads((output / "train.jsonl").read_text().splitlines()[0])
        self.assertEqual(row, {"filename": "walk.sps", "motion_path": "day1/walk.sps"})

    def test_csv_lookup_is_explicit(self):
        source, motion = self.prepare_motion_tree(".csv")
        output = self.root / "resolved"
        splits.resolve_splits(source, motion, output, ".csv")
        row = json.loads((output / "test.jsonl").read_text())
        self.assertEqual(row["motion_path"], "day1/jump.csv")

    def test_ambiguous_lookup_rejected(self):
        source, motion = self.prepare_motion_tree()
        (motion / "walk.sps").write_text("another file")
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            splits.resolve_splits(source, motion, self.root / "resolved")

    def test_missing_lookup_rejected(self):
        source, motion = self.prepare_motion_tree()
        (motion / "day1/jump.sps").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            splits.resolve_splits(source, motion, self.root / "resolved")

    def test_filename_schema_rejects_extra_content(self):
        source, _ = self.prepare_motion_tree()
        (source / "test.jsonl").write_text('{"filename":"jump.sps","caption":"not allowed"}\n')
        with self.assertRaisesRegex(ValueError, "Only the filename"):
            splits.read_splits(source)

    def test_traversal_rejected(self):
        for name in ("../walk.sps", "/walk.sps", "dir\\walk.sps", "walk\nsps", ""):
            with self.subTest(name=name), self.assertRaises(ValueError):
                splits.valid_name(name)


if __name__ == "__main__":
    unittest.main()
