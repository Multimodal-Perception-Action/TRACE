#!/usr/bin/env python3
"""将本包 SPS 渲染为 MP4、封面和本地预览页，不重新运行生成模型。"""
import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
PRODUCTS = ('motion_smoothed.mp4', 'preview.png', 'index.html', 'render_report.json', 'render_timing.json')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_motion(path):
    """使用生成器相同的坐标转换，严格检查帧号、帧率和关节完整性。"""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from trace_core.motion_io import SPS_JOINTS, parse_sps_file, convert_to_35dim
    text = path.read_text(encoding='utf-8')
    match = re.search(r'^# input_fps:\s*(\d+)\s*$', text, re.M)
    if not match or int(match.group(1)) <= 0:
        raise ValueError('SPS 缺少有效的 # input_fps 标记；请使用本包导出的 SPS')
    blocks = re.split(r'^\s*(\d+)\s*$', text, flags=re.M)
    indices = [int(i) for i in blocks[1::2]]
    if not indices or indices != list(range(len(indices))):
        raise ValueError('SPS 帧号必须从 0 开始连续递增')
    for body in blocks[2::2]:
        names = [line.split()[0] for line in body.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if len(names) != len(SPS_JOINTS) or set(names) != set(SPS_JOINTS):
            raise ValueError('SPS 中存在缺失或重复关节')
    pose = convert_to_35dim(parse_sps_file(path))
    if pose.shape != (len(indices), 35) or not np.isfinite(pose).all():
        raise ValueError('SPS 姿态形状或数值无效')
    qpos = np.concatenate((pose[:, [2, 0, 1]]*.01,
        Rotation.from_euler('xyz', pose[:, 3:6], degrees=True).as_quat(scalar_first=True),
        np.deg2rad(pose[:, 6:])), axis=1)
    return qpos, int(match.group(1))


def scene_xml(width, height):
    """只增加场景外观，不改变机器人运动学、姿态或关节限位。"""
    folder = ROOT/'assets/unitree_g1'
    tree = ET.parse(folder/'g1_mocap_29dof.xml')
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(folder/'meshes'))
    # 原始 XML 带有演示用场景块；先替换这些外观节点，避免重复地面和 skybox。
    for node in root.findall('visual') + root.findall('statistic'):
        root.remove(node)
    for node in root.findall('asset'):
        for texture in list(node.findall('texture')):
            if texture.get('type') == 'skybox':
                node.remove(texture)
    for node in root.findall('worldbody'):
        for child in list(node):
            if child.tag == 'light' or (child.tag == 'geom' and child.get('type') == 'plane'):
                node.remove(child)
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, 'quality', shadowsize='2048', offsamples='4')
    ET.SubElement(visual, 'headlight', ambient='0.4 0.4 0.4', diffuse='0.7 0.7 0.7', specular='0.15 0.15 0.15')
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient',
        rgb1='0.85 0.9 0.96', rgb2='0.98 0.99 1', width='512', height='3072')
    ET.SubElement(asset, 'texture', name='floor_tex', type='2d', builtin='checker',
        rgb1='0.66 0.76 0.86', rgb2='0.83 0.89 0.95', width='512', height='512')
    ET.SubElement(asset, 'material', name='floor_mat', texture='floor_tex',
        texrepeat='2 2', texuniform='true', reflectance='0.08', shininess='0.15')
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 0.1',
        pos='0 0 0', material='floor_mat', contype='0', conaffinity='0')
    ET.SubElement(world, 'light', name='key', pos='2 -3 5', dir='-0.4 0.5 -1',
        directional='true', castshadow='true', diffuse='0.7 0.7 0.7')
    return ET.tostring(root, encoding='unicode')


def preview_html(prompt, frames, fps):
    safe = html.escape(prompt or 'G1 动作可视化')
    return f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACE · 动作预览</title><style>
body{{margin:0;background:#edf2f7;color:#182c43;font:16px system-ui,sans-serif}}
main{{max-width:1060px;margin:36px auto;padding:24px}}h1{{font-size:28px}}
.card{{background:white;border-radius:18px;overflow:hidden;box-shadow:0 12px 36px #18344a18}}
video{{width:100%;display:block;background:#dfe8f3}}.info{{padding:20px 24px}}
.muted{{color:#54677c;font-size:14px}}button,a{{color:#0b677e}}button{{padding:8px 16px;cursor:pointer}}
</style><main><h1>TRACE · G1 动作预览</h1><div class="card">
<video id="motion" controls playsinline preload="metadata" poster="preview.png">
<source src="motion_smoothed.mp4" type="video/mp4">请用现代浏览器打开本页。</video>
<div class="info"><p>{safe}</p><p class="muted">{frames} 帧 · {fps} FPS · {frames/fps:.2f} 秒 · 平滑 SPS 的运动学回放</p>
<button onclick="document.getElementById('motion').currentTime=0;document.getElementById('motion').play()">重新播放</button>
<a href="motion_smoothed.mp4" download>下载视频</a>
<p class="muted">固定镜头、原始帧率；未执行 IK、关节裁剪或物理仿真。地板高度为世界坐标 z=0。</p>
</div></div></main></html>'''


def render_motion(source, output, backend='egl', render_device=0, width=960, height=720, prompt='', camera_reference=None):
    """输出可视化文件；已存在的同名文件会报错，不覆盖源动作。"""
    if width < 64 or height < 64 or width > 4096 or height > 4096 or width % 2 or height % 2:
        raise ValueError('视频宽高需为 [64,4096] 范围内的偶数')
    if backend not in ('egl', 'osmesa') or render_device < 0:
        raise ValueError('渲染后端需为 egl/osmesa，渲染设备编号不能为负数')
    source, output = Path(source).resolve(), Path(output).resolve()
    qpos, fps = load_motion(source)
    for name in PRODUCTS:
        if (output/name).exists():
            raise FileExistsError(f'可视化文件已存在，请改用新输出目录：{output/name}')
    output.mkdir(parents=True, exist_ok=True)
    # 必须在首次导入 MuJoCo / OpenGL 前选择后端；独立进程与推理显存隔离。
    os.environ['MUJOCO_GL'] = backend
    os.environ['PYOPENGL_PLATFORM'] = backend
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(render_device)
    import mujoco as mj
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from trace_core.motion_io import SPS_JOINTS
    start = time.perf_counter()
    model = mj.MjModel.from_xml_string(scene_xml(width, height))
    data = mj.MjData(model)
    ids = np.array([mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in SPS_JOINTS[1:]])
    if np.any(ids < 0):
        raise ValueError('随包机器人模型缺少必要关节')
    addresses = model.jnt_qposadr[ids]

    def set_frame(frame):
        data.qpos[:7] = frame[:7]
        data.qpos[addresses] = frame[7:]
        data.qvel[:] = 0
        # FK 回放，不调用 mj_step，也不夹紧超限关节。
        mj.mj_forward(model, data)

    # 用全序列身体包围盒确定固定镜头；不逐帧追踪，以便观察真实位移和滑步。
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    framing_qpos = load_motion(Path(camera_reference).resolve())[0] if camera_reference else qpos
    for frame in framing_qpos:
        set_frame(frame)
        low = np.minimum(low, data.xpos[1:].min(axis=0))
        high = np.maximum(high, data.xpos[1:].max(axis=0))
    low -= .23
    high += .23
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.lookat[:] = (low+high)/2
    half_fov = math.radians(float(model.vis.global_.fovy)/2)
    fit_fov = min(half_fov, math.atan(math.tan(half_fov)*width/height))
    camera.distance = max(2.5, np.linalg.norm(high-low)/2 / math.sin(fit_fov)*1.08)
    camera.azimuth, camera.elevation = 135, -16
    options = mj.MjvOption()
    options.geomgroup[0] = 0  # 隐藏碰撞代理，保留外观网格。
    # 地面需单独放入可见组，避免被上一行隐藏。
    model.geom_group[mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, 'floor')] = 2
    temporary = output/'.motion_smoothed.partial.mp4'
    renderer = mj.Renderer(model, height=height, width=width)
    try:
        with imageio.get_writer(str(temporary), fps=fps, codec='libx264', pixelformat='yuv420p',
                macro_block_size=1, quality=8, ffmpeg_log_level='error',
                output_params=['-movflags', '+faststart']) as writer:
            for i, frame in enumerate(qpos):
                set_frame(frame)
                renderer.update_scene(data, camera=camera, scene_option=options)
                rgb = renderer.render()
                writer.append_data(rgb)
                if i == len(qpos)//2:
                    imageio.imwrite(output/'preview.png', rgb)
    finally:
        renderer.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if count != len(qpos) or abs(duration-len(qpos)/fps) > 1/fps+.01:
        raise RuntimeError('视频帧数或时长校验失败，请检查 FFmpeg')
    temporary.replace(output/'motion_smoothed.mp4')
    (output/'index.html').write_text(preview_html(prompt, len(qpos), fps), encoding='utf-8')
    report = dict(success=True, source_sps=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        frames=len(qpos), fps=fps, duration_seconds=duration, width=width, height=height,
        backend=backend, egl_device=render_device if backend == 'egl' else None,
        mujoco_version=mj.__version__, camera='fixed_full_trajectory',
        camera_reference=str(Path(camera_reference).resolve()) if camera_reference else str(source),
        ground_height_m=0, joint_clipping=False, ik=False, physics_simulation=False,
        video='motion_smoothed.mp4', preview='index.html', seconds=time.perf_counter()-start)
    save(output/'render_report.json', report)
    save(output/'render_timing.json', dict(stage='render', seconds=report['seconds'], backend=backend))
    print(f"可视化完成：{output/'motion_smoothed.mp4'}\n预览网页：{output/'index.html'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='本包导出的 SPS 文件')
    parser.add_argument('--output', required=True, type=Path, help='视频输出目录，可与 SPS 目录相同')
    parser.add_argument('--backend', choices=['egl', 'osmesa'], default='egl')
    parser.add_argument('--render-device', type=int, default=0, help='EGL 设备编号；独立于 PyTorch 的设备编号')
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--prompt', default='')
    parser.add_argument('--camera-reference', type=Path, help='对比渲染共用的取景参考 SPS，不改变播放动作')
    args = parser.parse_args()
    try:
        render_motion(args.input, args.output, args.backend, args.render_device, args.width, args.height, args.prompt, args.camera_reference)
    except Exception as exc:
        parser.exit(1, f'可视化失败：{exc}\n已有 SPS 保留。请检查渲染依赖及 EGL 驱动；修复后可单独重跑 render.py。\n')


if __name__ == '__main__':
    main()
