"""Keep the review cohort identical to the original survey prompts and order."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_user_study_prompts import load_cases
from tools.run_readme_test_prompts import generation_command
from trace_core.references import sha256


class UserStudyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.survey = Path(self.tmp.name)/"survey.json"
        self.comparison = Path(self.tmp.name)/"comparison.json"
        self.rows = [
            dict(uid=f"sample_{i:03d}", kind="standard", text=f"Prompt {i}", zh="")
            for i in range(10)
        ]
        self.rows += [dict(uid="wild_01", kind="wild", text="Not requested")]
        self.survey.write_text(json.dumps(self.rows))
        self.archive = dict(
            cohort="original_user_study_standard_10",
            survey_source_sha256=sha256(self.survey),
            samples=[
                dict(sample_id=row["uid"], text=row["text"],
                     uid=f"motion:{i}", survey_order=i+1)
                for i, row in enumerate(self.rows[:10])
            ],
        )
        self.comparison.write_text(json.dumps(self.archive))

    def test_keeps_all_test_prompts_in_order_and_excludes_wild(self):
        cases = load_cases(self.survey, self.comparison)
        self.assertEqual(len(cases), 10)
        self.assertEqual([c["id"] for c in cases], [r["uid"] for r in self.rows[:10]])
        self.assertEqual([c["text"] for c in cases], [r["text"] for r in self.rows[:10]])

    def test_rejects_changed_survey(self):
        self.rows[0]["text"] = "Different prompt"
        self.survey.write_text(json.dumps(self.rows))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_cases(self.survey, self.comparison)

    def test_rejects_prompt_mismatch(self):
        self.archive["samples"][2]["text"] = "Different prompt"
        self.comparison.write_text(json.dumps(self.archive))
        with self.assertRaisesRegex(ValueError, "mismatch"):
            load_cases(self.survey, self.comparison)

    def test_rejects_missing_case(self):
        self.archive["samples"].pop()
        self.comparison.write_text(json.dumps(self.archive))
        with self.assertRaisesRegex(ValueError, "exactly ten"):
            load_cases(self.survey, self.comparison)

    def test_free_mode_has_no_pose_arguments(self):
        command = generation_command(Path("/tmp/review"), dict(id="test", text="Walk"), 2, "free")
        self.assertEqual(command[command.index("--boundary")+1], "free")
        self.assertNotIn("--start-ref", command)
        self.assertNotIn("--end-ref", command)
        self.assertFalse(any("idle_pose" in arg for arg in command))

    def test_standing_mode_keeps_both_references(self):
        command = generation_command(Path("/tmp/review"), dict(id="test", text="Walk"), 2)
        self.assertEqual(command[command.index("--boundary")+1], "standing")
        self.assertEqual(command[command.index("--start-ref")+1], "/tmp/review/idle.sps")
        self.assertEqual(command[command.index("--end-ref")+1], "/tmp/review/idle.sps")

    def test_refuses_to_swap_to_original_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "same model"):
            generation_command(Path("/tmp/review"), dict(id="test", text="Walk"), 2, "none")


if __name__ == "__main__":
    unittest.main()
