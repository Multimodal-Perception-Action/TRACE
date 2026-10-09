"""可视化输入、单位转换和输出保护测试；不要求 OpenGL 上下文。"""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from render import load_motion, preview_html, render_motion, scene_xml, ROOT
from trace_core.motion_io import write_sps


class RenderTests(unittest.TestCase):
    def test_coordinate_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'motion.sps'
            pose = np.zeros((3, 35), dtype=np.float32)
            pose[:, :3] = [100, 200, 300]
            pose[:, 5] = 90
            pose[:, 6] = 30
            write_sps(path, pose, fps=20)
            q, fps = load_motion(path)
            self.assertEqual(fps, 20)
            np.testing.assert_allclose(q[:, :3], [[3, 1, 2]]*3)
            np.testing.assert_allclose(q[:, 3:7], [[2**-.5, 0, 0, 2**-.5]]*3, atol=1e-6)
            np.testing.assert_allclose(q[:, 7], np.pi/6)

    def test_incomplete_sps_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bad.sps'
            path.write_text('skel_poseset_v01\n# input_fps: 20\n0\nroot ty:80\n')
            with self.assertRaisesRegex(ValueError, '关节'):
                load_motion(path)

    def test_output_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'motion.sps'
            write_sps(path, np.zeros((1, 35)), fps=20)
            video = Path(tmp)/'motion_smoothed.mp4'
            video.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                render_motion(path, tmp)
            self.assertEqual(video.read_bytes(), b'preserve')

    def test_prompt_escaped(self):
        page = preview_html('<script>alert(1)</script>', 20, 20)
        self.assertNotIn('<script>', page)
        self.assertIn('&lt;script&gt;', page)

    def test_packaged_meshes(self):
        scene = ET.fromstring(scene_xml(960, 720))
        self.assertEqual(scene.find('visual/global').get('offwidth'), '960')
        for mesh in scene.findall('asset/mesh'):
            self.assertTrue((ROOT/'assets/unitree_g1/meshes'/mesh.get('file')).is_file())


if __name__ == '__main__':
    unittest.main()
