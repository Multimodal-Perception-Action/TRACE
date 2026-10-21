"""Check that the dual-robot visual scene preserves the original kinematics."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from render import scene_xml
from tools.render_edit_overlay import overlay_xml, composite_priority, colors
from tools import render_edit_overlay
from trace_core.motion_io import SPS_JOINTS


class OverlayTests(unittest.TestCase):
    def test_single_edit_cli_without_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "jump_edit"
            directory.mkdir()
            (directory / "request.json").write_text(json.dumps({"instruction": "Can you jump higher?"}))
            (directory / "result.json").write_text(json.dumps({
                "method": "experimental_generator_reference_resampling",
                "numerical_guidance": {"gain_cm": 14}}))
            output = Path(temporary) / "overlay"
            argv = ["render_edit_overlay.py", "--edit", str(directory), "--output", str(output)]
            with patch.object(sys, "argv", argv), patch.object(render_edit_overlay, "render_case") as render, patch.object(render_edit_overlay, "compose") as compose:
                render_edit_overlay.main()
            self.assertEqual(render.call_args.kwargs["edit_directory"], directory)
            self.assertEqual(render.call_args.args[1]["instruction"], "Can you jump higher?")
            self.assertIn("height target", render.call_args.args[1]["method_label"])
            compose.assert_called_once()

    def test_before_priority_colors(self):
        before, after = colors("before")
        self.assertEqual(before[3], 1)
        self.assertLess(after[3], 1)

    def test_overlap_keeps_primary_pixels(self):
        combined = np.full((2, 2, 3), 200, np.uint8)
        foreground = np.full((2, 2, 3), 20, np.uint8)
        geom = int(mj.mjtObj.mjOBJ_GEOM)
        segmentation = np.array([[[5, geom], [8, geom]], [[5, -1], [-1, -1]]])
        actual = composite_priority(combined, foreground, segmentation, [5], geom)
        np.testing.assert_array_equal(actual[0, 0], foreground[0, 0])
        np.testing.assert_array_equal(actual[0, 1], combined[0, 1])
        np.testing.assert_array_equal(actual[1], combined[1])

    def test_instances_are_named_and_distinct(self):
        root = ET.fromstring(overlay_xml())
        self.assertEqual(len(root.findall("worldbody/body")), 2)
        self.assertEqual(
            [x.get("name") for x in root.findall(".//freejoint")],
            ["before_pelvis", "after_pelvis"],
        )
        self.assertIsNone(root.find("actuator"))

    def test_both_copies_match_original_fk(self):
        original = mj.MjModel.from_xml_string(scene_xml(640, 640))
        combined = mj.MjModel.from_xml_string(overlay_xml())
        reference, overlay = mj.MjData(original), mj.MjData(combined)
        for version, prefix in enumerate(("before", "after")):
            root_pose = np.array([.3, -.1, .8 + .1 * version, 1., 0., 0., 0.])
            angles = np.linspace(-.15, .35, 29) + .1 * version
            free = mj.mj_name2id(combined, mj.mjtObj.mjOBJ_JOINT, prefix + "_pelvis")
            offset = combined.jnt_qposadr[free]
            reference.qpos[:7] = root_pose
            overlay.qpos[offset:offset + 7] = root_pose
            for name, angle in zip(SPS_JOINTS[1:], angles):
                a = mj.mj_name2id(original, mj.mjtObj.mjOBJ_JOINT, name)
                b = mj.mj_name2id(combined, mj.mjtObj.mjOBJ_JOINT, prefix + "_" + name)
                reference.qpos[original.jnt_qposadr[a]] = angle
                overlay.qpos[combined.jnt_qposadr[b]] = angle
            mj.mj_forward(original, reference)
            mj.mj_forward(combined, overlay)
            for body in range(1, original.nbody):
                name = mj.mj_id2name(original, mj.mjtObj.mjOBJ_BODY, body)
                other = mj.mj_name2id(combined, mj.mjtObj.mjOBJ_BODY, prefix + "_" + name)
                self.assertGreater(other, 0)
                np.testing.assert_allclose(reference.xpos[body], overlay.xpos[other], atol=1e-12)
                np.testing.assert_allclose(reference.xmat[body], overlay.xmat[other], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
