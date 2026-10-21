#!/usr/bin/env python3
"""一键端到端推理：原始文本 → LLM Plan → 连续动作 → 平滑 SPS → 可视化。"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def check_models(models, boundary='standing', generator_checkpoint=None, generation_stats=None,
                 custom_both_references=False):
    models = Path(models).resolve()
    required = ['planner_adapter/adapter_model.safetensors',
        'planner_adapter/adapter_config.json']
    if boundary == 'standing' and not custom_both_references:
        required.append('standing_pose35.npy')
    required.extend([str(Path(generator_checkpoint).resolve() if generator_checkpoint else
                         models/('generator_boundary.pt' if boundary in ('standing', 'free') else 'generator.pt')),
                     str(Path(generation_stats).resolve() if generation_stats else models/'generation_stats.npz')])
    for name in ('planner_base', 'text_embedding'):
        required.extend([f'{name}/config.json', f'{name}/tokenizer.json', f'{name}/tokenizer_config.json',
            f'{name}/model.safetensors.index.json'])
        index = models/name/'model.safetensors.index.json'
        if index.is_file():
            required.extend(f'{name}/{s}' for s in set(json.loads(index.read_text())['weight_map'].values()))
    missing = [p for p in required if not (models/p).is_file()]
    if missing:
        raise FileNotFoundError('模型文件缺失：\n'+'\n'.join(missing))


def generate(text, output, models=None, device='cuda:0', seed=42, steps=50, cfg=2.5,
             precision='auto', root_hz=3., joint_hz=4., filter_order=4, cpu_threads=4,
             render=True, render_backend='egl', render_device=0, boundary='standing', boundary_frames=8,
             contact_refine='heel_toe', generator_checkpoint=None, generation_stats=None,
             start_ref=None, end_ref=None, start_frame=0, end_frame=0):
    """Python 工程接口；串行启动子进程，保证每阶段结束后完整释放 GPU 内存。"""
    import torch
    if contact_refine not in ('heel_toe','joint','none'):
        raise ValueError('contact_refine 为 heel_toe/joint/none')
    if contact_refine!='none' and importlib.util.find_spec('mujoco') is None:
        raise RuntimeError('联合修复需要 mujoco；请安装 requirements-render.txt，或 --contact-refine none')
    if boundary not in ('standing', 'free', 'none') or not 1 <= boundary_frames <= 16:
        raise ValueError('boundary must be standing/free/none; boundary_frames must be 1–16')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('输入文本不能为空')
    if not 1 <= steps <= 1000 or not 0 <= cfg <= 20:
        raise ValueError('steps 需位于 [1,1000]，cfg 需位于 [0,20]')
    if not 0 <= seed < 2**32 or cpu_threads < 1:
        raise ValueError('seed 需位于 [0,2^32)，cpu_threads 至少为 1')
    if not 0 < root_hz < 10 or not 0 < joint_hz < 10 or not 1 <= filter_order <= 8:
        raise ValueError('平滑截止频率需位于 (0,10)，阶数需位于 [1,8]')
    if render:
        missing = [n for n in ('mujoco', 'imageio', 'imageio_ffmpeg', 'PIL') if importlib.util.find_spec(n) is None]
        if missing:
            raise RuntimeError('缺少渲染依赖：'+', '.join(missing)+'；请安装 requirements-render.txt，或使用 --skip-render')
        if render_backend not in ('egl', 'osmesa') or render_device < 0:
            raise ValueError('渲染后端或设备编号无效')
    dev = torch.device(device)
    if dev.type not in ('cpu', 'cuda'):
        raise ValueError('当前交接版支持 CUDA 或 CPU')
    if dev.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('未检测到 CUDA，请检查驱动与 PyTorch；可用 --device cpu 做慢速调试')
        torch.cuda.set_device(dev)
    if precision == 'auto':
        precision = 'bf16' if dev.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if precision == 'bf16' and (dev.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('此设备不支持 BF16，请改用 --precision fp32')
    output, models = Path(output).resolve(), Path(models or ROOT/'models').resolve()
    if boundary != 'standing' and (start_ref or end_ref):
        raise ValueError('Pose references require --boundary standing')
    generator_checkpoint = Path(generator_checkpoint or models/(
        'generator_boundary.pt' if boundary in ('standing', 'free') else 'generator.pt')).resolve()
    generation_stats = Path(generation_stats or models/'generation_stats.npz').resolve()
    check_models(models, boundary, generator_checkpoint, generation_stats, bool(start_ref and end_ref))
    from trace_core.references import prepare_references, validate_checkpoint, sha256
    checkpoint_info = validate_checkpoint(generator_checkpoint, boundary)
    refs = prepare_references(models/'standing_pose35.npy', start_ref, end_ref,
                              start_frame, end_frame) if boundary == 'standing' else None
    output.mkdir(parents=True, exist_ok=False)
    config = dict(text=text.strip(), models=str(models), device=str(dev), seed=seed, steps=steps,
        cfg=cfg, precision=precision, root_hz=root_hz, joint_hz=joint_hz,
        filter_order=filter_order, cpu_threads=cpu_threads, render=render,
        render_backend=render_backend, render_device=render_device,
        boundary=boundary, boundary_frames=boundary_frames, contact_refine=contact_refine,
        generator_checkpoint=str(generator_checkpoint), generation_stats=str(generation_stats))
    save(output/'checkpoint.json', checkpoint_info)
    if refs is not None:
        import numpy as np
        reference_path = output/'boundary_reference_pose35.npy'
        np.save(reference_path, refs, allow_pickle=False)
        config.update(boundary_reference_path=str(reference_path),
                      boundary_reference_sha256=sha256(reference_path))
        save(output/'boundary_reference_info.json', dict(
            start=dict(file=Path(start_ref).name if start_ref else 'default', frame=start_frame),
            end=dict(file=Path(end_ref).name if end_ref else 'default', frame=end_frame),
            sha256=config['boundary_reference_sha256'],
            semantics='Local idle posture; absolute world XY/yaw are not constrained'))
    save(output/'request.json', config)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'numpy', 'scipy', 'accelerate')}
    save(output/'environment.json', dict(python=sys.version, packages=versions,
        gpu=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else None))
    start = time.perf_counter()
    env = os.environ.copy()
    # 子进程只使用交付目录和当前虚拟环境，不继承历史工程的模块搜索路径。
    env.pop('PYTHONPATH', None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    try:
        for stage in ('planner', 'embedding', 'diffusion', 'smooth', 'contact'):
            print(f'\n[{stage}] 开始', flush=True)
            subprocess.run([sys.executable, str(ROOT/'run.py'), '--stage', stage,
                '--output', str(output)], env=env, check=True)
        if render:
            print('\n[render] 开始', flush=True)
            subprocess.run([sys.executable, str(ROOT/'render.py'),
                '--input', str(output/'motion_final.sps'), '--output', str(output),
                '--backend', render_backend, '--render-device', str(render_device),
                '--prompt', text.strip()], env=env, check=True)
    except Exception as exc:
        save(output/'FAILED.json', dict(error=str(exc), seconds=time.perf_counter()-start))
        raise
    plan = json.loads((output/'plan.json').read_text())
    timings = [json.loads((output/(s+'_timing.json')).read_text()) for s in ('planner', 'embedding', 'diffusion', 'smooth', 'contact')]
    if render:
        timings.append(json.loads((output/'render_timing.json').read_text()))
    result = dict(success=True, frames=plan['frames'], fps=20, duration_seconds=plan['frames']/20,
        seconds=time.perf_counter()-start, stages=timings, raw_sps='motion_raw.sps',
        smoothed_sps='motion_smoothed.sps', qpos='motion_smoothed_qpos.npz',
        final_sps='motion_final.sps', final_qpos='motion_final_qpos.npz',
        planner='LLM joint duration and events', gt_used=False, postprocessing=True, ik=contact_refine!='none',
        contact_refine=contact_refine,
        boundary=boundary, boundary_frames=boundary_frames if boundary=='standing' else 0,
        generator=generator_checkpoint.name, generator_sha256=checkpoint_info['sha256'],
        rendered=render, video='motion_smoothed.mp4' if render else None,
        preview='index.html' if render else None)
    save(output/'result.json', result)
    print(f"\n推理完成。最终 SPS：{output/'motion_final.sps'}", flush=True)
    return output/'motion_final.sps'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--text', help='原始动作描述，推荐英文')
    source.add_argument('--text-file', type=Path, help='UTF-8 文本文件')
    parser.add_argument('--output', type=Path, help='新建输出目录，已有目录不覆盖')
    parser.add_argument('--models', type=Path, default=ROOT/'models')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--boundary', choices=['standing','free','none'], default='standing',
                        help='standing: pose references; free: same boundary model without references; none: original generator')
    parser.add_argument('--boundary-frames', type=int, default=8, help='首尾各约束多少帧，默认 8（0.4 秒），范围 1–16')
    parser.add_argument('--generator-checkpoint', type=Path, help='Compatible TRACE inference checkpoint')
    parser.add_argument('--generation-stats', type=Path, help='Normalization statistics matched to the checkpoint')
    parser.add_argument('--start-ref', type=Path, help='Start idle pose: G1 SPS or pose35 NPY')
    parser.add_argument('--end-ref', type=Path, help='End idle pose, independent of --start-ref')
    parser.add_argument('--start-frame', type=int, default=0, help='Zero-based pose index in the start reference')
    parser.add_argument('--end-frame', type=int, default=0, help='Zero-based pose index in the end reference')
    parser.add_argument('--contact-refine', choices=['heel_toe','joint','none'], default='heel_toe', help='默认前后掌标签＋四点联合修复；joint 为旧中心约束；none 仅平滑')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=2.5)
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp32'], default='auto')
    parser.add_argument('--root-hz', type=float, default=3.)
    parser.add_argument('--joint-hz', type=float, default=4.)
    parser.add_argument('--filter-order', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--skip-render', action='store_true', help='只生成平滑 SPS，不渲染视频')
    parser.add_argument('--render-backend', choices=['egl', 'osmesa'], default='egl')
    parser.add_argument('--render-device', type=int, default=0, help='EGL 设备编号，默认 0；与 CUDA 编号独立')
    parser.add_argument('--check-models', action='store_true', help='检查权重文件是否齐全')
    parser.add_argument('--stage', choices=['planner', 'embedding', 'diffusion', 'smooth', 'contact'], help=argparse.SUPPRESS)
    a = parser.parse_args()
    if a.check_models:
        check_models(a.models.resolve(), a.boundary, a.generator_checkpoint, a.generation_stats,
                     bool(a.start_ref and a.end_ref))
        print('模型文件齐全')
        return
    if a.stage:
        from trace_core.pipeline import run_stage
        config = json.loads((a.output/'request.json').read_text())
        run_stage(a.stage, config, a.output)
        return
    if a.output is None or (a.text is None and a.text_file is None):
        parser.error('请提供 --text 或 --text-file，以及 --output')
    generate(a.text if a.text is not None else a.text_file.read_text(encoding='utf-8'), a.output,
        a.models, a.device, a.seed, a.steps, a.cfg, a.precision, a.root_hz, a.joint_hz, a.filter_order, a.cpu_threads,
        render=not a.skip_render, render_backend=a.render_backend, render_device=a.render_device,
        boundary=a.boundary, boundary_frames=a.boundary_frames, contact_refine=a.contact_refine,
        generator_checkpoint=a.generator_checkpoint, generation_stats=a.generation_stats,
        start_ref=a.start_ref, end_ref=a.end_ref, start_frame=a.start_frame, end_frame=a.end_frame)


if __name__ == '__main__':
    main()
