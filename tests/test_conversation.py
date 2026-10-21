"""Conversation state and edit compilation must not silently change intent."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import converse
from trace_core.conversation import parse_intent, compile_intent, source_context, atomic_json, validate_explicit_distance, validate_changed_caption
from trace_core.motion_io import write_sps


def intent(**changes):
    return dict(dict(action="jump_height", target_caption="A person performs a higher jumping jack.",
                     body="lower_body", direction="higher", magnitude="moderate", amount_cm=None,
                     time_region="full", reason="Increase the height of the current jump."), **changes)


def summary(height=10):
    return dict(frames=70, boundary_frames=8,
                airborne_segments=[dict(start=27, end=34, peak=30, clearance_cm=height)])


class ConversationTests(unittest.TestCase):
    def test_json_and_fenced_json(self):
        value = intent()
        self.assertEqual(parse_intent(json.dumps(value)), value)
        self.assertEqual(parse_intent('```json\n'+json.dumps(value)+'\n```'), value)

    def test_bad_schema_and_nonfinite_targets(self):
        for changes in (dict(amount_cm=True), dict(amount_cm=float('nan')),
                        dict(amount_cm=30), dict(body="right_hand"), dict(direction="none")):
            with self.assertRaises(ValueError):
                parse_intent(json.dumps(intent(**changes)))
        text = json.dumps(intent()).replace('"action":', '"action":"unsupported","action":', 1)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            parse_intent(text)

    def test_strength_defaults_are_explicit(self):
        result = compile_intent(intent(), summary())
        self.assertEqual(result["height_gain_cm"], 10)
        self.assertEqual(result["target_clearance_cm"], 20)
        self.assertIn("default", result["target_origin"])
        self.assertEqual(result["active_span"], [20, 48])

    def test_explicit_distance_cannot_be_invented_or_ignored(self):
        validate_explicit_distance(intent(), "Can you jump higher?")
        validate_explicit_distance(intent(amount_cm=5), "Jump five centimeters higher.")
        validate_explicit_distance(intent(amount_cm=10), "Jump 0.1 m higher.")
        with self.assertRaises(ValueError):
            validate_explicit_distance(intent(amount_cm=10), "Can you jump higher?")
        with self.assertRaises(ValueError):
            validate_explicit_distance(intent(), "Jump 10 cm higher.")
        with self.assertRaises(ValueError):
            validate_explicit_distance(intent(amount_cm=5), "Jump 10 cm higher.")

    def test_minor_rebound_does_not_replace_primary_jump(self):
        data = summary()
        data["airborne_segments"].append(dict(start=40, end=43, peak=41, clearance_cm=3))
        result = compile_intent(intent(), data)
        self.assertEqual(result["selected_source_apex"], 30)

    def test_style_edits_cannot_forward_an_unchanged_caption(self):
        context = dict(latest_caption="A person kicks forward with the right leg.")
        value = intent(action="body_style", direction="none", body="right_leg",
                       target_caption="A person kicks forward with the right leg!")
        with self.assertRaisesRegex(ValueError, "repeats the source"):
            validate_changed_caption(value, context)
        value["target_caption"] = "A person kicks forward higher with the right leg."
        validate_changed_caption(value, context)
        validate_changed_caption(intent(), dict(latest_caption=intent()["target_caption"]))

    def test_next_turn_is_relative_to_current_measurement(self):
        result = compile_intent(intent(direction="lower", magnitude="slight"), summary(20))
        self.assertEqual(result["height_gain_cm"], -5)
        self.assertEqual(result["target_clearance_cm"], 15)
        result = compile_intent(intent(amount_cm=3), summary(15))
        self.assertEqual(result["target_clearance_cm"], 18)
        self.assertIn("explicit", result["target_origin"])

    def test_rejects_ambiguous_or_unsupported_requests(self):
        with self.assertRaisesRegex(ValueError, "unsupported capability"):
            compile_intent(intent(action="unsupported"), summary())
        for events in ([], summary()["airborne_segments"]*2):
            data = dict(summary(), airborne_segments=events)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                compile_intent(intent(), data)

    def test_bounds_and_non_jump_scope(self):
        with self.assertRaisesRegex(ValueError, "2..45"):
            compile_intent(intent(direction="lower"), summary(5))
        with self.assertRaisesRegex(ValueError, "insufficient room"):
            compile_intent(intent(time_region="first_half"), summary())
        result = compile_intent(intent(action="body_style", direction="none", body="torso"), summary())
        self.assertIsNone(result["height_gain_cm"])
        self.assertEqual(result["body"], "torso")

    def test_source_metadata_and_import_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = np.zeros((70, 35), dtype=np.float32)
            write_sps(root/"motion_final.sps", poses, fps=20)
            with self.assertRaisesRegex(ValueError, "source-text"):
                source_context(root)
            atomic_json(root/"request.json", dict(text="Perform a jumping jack."))
            path, caption, _, recovered = source_context(root)
            self.assertEqual(path.name, "motion_final.sps")
            self.assertEqual(caption, "Perform a jumping jack.")
            np.testing.assert_array_equal(poses, recovered)

    def test_bundled_paper_demo_source(self):
        from trace_core.references import sha256
        root = Path(__file__).resolve().parents[1]/"examples/motions/jumping_jack"
        path, caption, request, poses = source_context(root/"source.sps")
        self.assertEqual(caption, "Perform a jumping jack.")
        self.assertEqual(sha256(path), request["source_sha256"])
        self.assertEqual(poses.shape, (44, 35))
        self.assertIsNone(request["test_split_id"])

    def test_state_advances_only_after_completed_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"source"
            session = root/"session"
            source.mkdir()
            session.mkdir()
            checkpoint = root/"checkpoint.pt"
            stats = root/"stats.npz"
            checkpoint.write_bytes(b"test-checkpoint")
            stats.write_bytes(b"test-stats")
            poses = np.zeros((70, 35), np.float32)
            poses[:, 1] = 80
            write_sps(source/"motion_final.sps", poses, fps=20)
            atomic_json(source/"request.json", dict(text="Perform a jumping jack.",
                        generator_checkpoint=str(checkpoint), generation_stats=str(stats)))
            args = SimpleNamespace(source=source, source_text=None, models=root,
                generator_checkpoint=None, device="cpu", skip_render=True, render_device=0,
                instruction="Can you jump higher?")
            sources = []
            def execute(command, environment):
                if "--stage" in command:
                    context = Path(command[command.index("--context")+1])
                    atomic_json(context.parent/"intent.json", intent())
                    atomic_json(context.parent/"compiled_edit.json", compile_intent(intent(), summary()))
                else:
                    source_path = Path(command[command.index("--source")+1])
                    sources.append(source_path)
                    output = Path(command[command.index("--output")+1])
                    output.mkdir()
                    write_sps(output/"motion_edited.sps", poses, fps=20)
                    atomic_json(output/"result.json", dict(protected_encoded_coordinates_exact=True, idle_bands_exact=True))
            with patch("trace_core.references.validate_checkpoint"), patch.object(converse, "summarize_motion", return_value=summary()), patch.object(converse, "execute", side_effect=execute):
                converse.run_turn(args, session, session/"session.json")
                args.source = None
                args.instruction = "A little lower, please."
                converse.run_turn(args, session, session/"session.json")
            self.assertEqual(sources[0], session/"initial.sps")
            self.assertEqual(sources[1], session/"turns/001/edit/motion_edited.sps")
            before = json.loads((session/"session.json").read_text())
            self.assertEqual(len(before["history"]), 2)
            with patch.object(converse, "summarize_motion", return_value=summary()), patch.object(converse, "execute", side_effect=ValueError("unsupported")):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    converse.run_turn(args, session, session/"session.json")
            self.assertEqual(json.loads((session/"session.json").read_text()), before)
            self.assertTrue((session/"turns/003/FAILED.json").exists())


if __name__ == "__main__":
    unittest.main()
