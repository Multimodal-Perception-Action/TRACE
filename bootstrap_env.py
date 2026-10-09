#!/usr/bin/env python3
"""无 sudo、无 ensurepip 的虚拟环境引导；可补齐安装失败后留下的环境。"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parent
PIP_WHEEL = ROOT/'bootstrap/pip-26.2.1-py3-none-any.whl'
PIP_SHA256 = '71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e'


def check_wheel(path=PIP_WHEEL):
    """只执行随包附带的官方固定版本，不在线下载并执行不固定的引导脚本。"""
    if not path.is_file():
        raise FileNotFoundError(f'缺少引导文件：{path}；请完整复制交接目录中的 bootstrap/')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PIP_SHA256:
        raise ValueError('pip 引导文件校验失败，请重新复制完整交接包，不要执行被修改的文件')


def ensure_environment(target):
    """重建缺失的环境脚本但不清空目录、不删除已有包、不修改系统环境。"""
    target = Path(target).expanduser().absolute()
    if target.exists() and any(target.iterdir()):
        cfg = target/'pyvenv.cfg'
        if not cfg.is_file():
            raise ValueError(f'{target} 是非空目录且不是虚拟环境，请设置新的 TRACE_VENV 路径')
        fields = dict(line.split('=', 1) for line in cfg.read_text().splitlines() if '=' in line)
        fields = {k.strip(): v.strip() for k, v in fields.items()}
        old_version = fields.get('version', '').split('.')[:2]
        current = [str(v) for v in sys.version_info[:2]]
        if old_version and old_version != current:
            raise ValueError(f'现有环境 Python 版本不同：{fields.get("version")}；请另设 TRACE_VENV，不会覆盖旧环境')

    # 不调用 ensurepip；适配发行版把它拆分到 python3.x-venv 包的情况。
    venv.EnvBuilder(with_pip=False, clear=False, symlinks=True).create(target)
    python = target/'bin/python'
    check = ('import os,sys; '
        'assert sys.prefix != sys.base_prefix, "拒绝修改系统 Python"; '
        'assert os.path.realpath(sys.prefix)==os.path.realpath(sys.argv[1]), "虚拟环境路径不匹配"')
    subprocess.run([str(python), '-I', '-c', check, str(target)], check=True)
    probe = subprocess.run([str(python), '-I', '-m', 'pip', '--version'], capture_output=True, text=True)
    if probe.returncode == 0:
        print(f'复用已有 pip：{probe.stdout.strip()}', flush=True)
        return python

    check_wheel()
    print('环境缺少 pip：使用随包官方 wheel 离线初始化，无需 sudo。', flush=True)
    # 隔离解释器并只把已校验 wheel 加入模块路径；安装目标仍是当前虚拟环境。
    code = ('import runpy,sys; wheel=sys.argv[1]; sys.path.insert(0,wheel); '
        'sys.argv=["pip","--isolated","--disable-pip-version-check","--no-cache-dir","install",'
        '"--no-index","--no-deps","--ignore-installed","--prefix",sys.prefix,wheel]; '
        'runpy.run_module("pip",run_name="__main__")')
    env = os.environ.copy()
    for key in tuple(env):
        # 防止外部 pip 配置把安装重定向到系统目录、用户目录或别的环境。
        if key.startswith('PIP_') or key in ('PYTHONPATH', 'PYTHONHOME'):
            env.pop(key)
    subprocess.run([str(python), '-I', '-c', code, str(PIP_WHEEL)], env=env, check=True)
    subprocess.run([str(python), '-I', '-m', 'pip', '--version'], check=True)
    return python


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, required=True)
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 11):
        parser.error('请使用 Python 3.10 或 3.11，并设置 TRACE_PYTHON 为该解释器')
    try:
        ensure_environment(args.venv)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'环境初始化失败：{exc}\n')
