"""回归检查缺少 ensurepip 时的安装引导，以及依赖不完整时的提前拦截。"""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from bootstrap_env import ROOT, check_wheel, ensure_environment


class BootstrapTests(unittest.TestCase):
    def test_official_wheel_digest(self):
        check_wheel()

    def test_modified_wheel_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp)/'broken.whl'
            wheel.write_bytes(b'not the official wheel')
            with self.assertRaisesRegex(ValueError, '校验失败'):
                check_wheel(wheel)

    def test_non_venv_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp)/'keep.txt'
            sentinel.write_text('需要保留的数据')
            with self.assertRaisesRegex(ValueError, '非空目录'):
                ensure_environment(tmp)
            self.assertEqual(sentinel.read_text(), '需要保留的数据')

    def test_mismatched_version_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'pyvenv.cfg'
            config.write_text('version = 0.1.2\n')
            with self.assertRaisesRegex(ValueError, '版本不同'):
                ensure_environment(tmp)
            self.assertEqual(config.read_text(), 'version = 0.1.2\n')

    def test_bootstrap_repeat_and_run_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'env'
            # 始终以 with_pip=False 创建，因此不依赖测试机是否安装 ensurepip。
            python = ensure_environment(target)
            version = subprocess.check_output([str(python), '-I', '-m', 'pip', '--version'], text=True)
            self.assertIn('pip 26.2.1', version)
            sentinel = target/'keep.txt'
            sentinel.write_text('重复安装时应保留')
            ensure_environment(target)
            self.assertEqual(sentinel.read_text(), '重复安装时应保留')
            self.assertTrue((target/'bin/activate').is_file())
            env = os.environ.copy()
            env.pop('TRACE_PYTHON', None)
            env['TRACE_VENV'] = str(target)
            output = Path(tmp)/'must_not_be_created'
            result = subprocess.run(['bash', str(ROOT/'run.sh'), '--text', 'Perform a jumping jack.',
                '--output', str(output)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('环境依赖未安装完整', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
