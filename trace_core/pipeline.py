"""文本、Plan、动作和 SPS 的单阶段实现；由父进程分阶段调用以释放显存。"""
import json
import time
from pathlib import Path
import numpy as np
import torch


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def language_dtype(device):
    if device.type == 'cpu':
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def predict_plan(config, output, device):
    """联合 LLM 预测时长与全部事件；不使用 GT，也不回退到小预测器。"""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from .llm_plan import SYSTEM
    from .llm_compile import canonical_parse, compile_response
    models = Path(config['models'])
    tokenizer = AutoTokenizer.from_pretrained(models/'planner_base', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(models/'planner_base',
        torch_dtype=language_dtype(device), attn_implementation='sdpa', local_files_only=True).to(device)
    model = PeftModel.from_pretrained(model, models/'planner_adapter', local_files_only=True).eval()
    prompt = tokenizer.apply_chat_template([
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': config['text']}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokens = tokenizer(prompt, return_tensors='pt').to(device)
    if tokens['input_ids'].shape[1] > 512:
        raise ValueError('指令超出语言前端 512 token 输入预算，请缩短描述；不静默截断')
    with torch.inference_mode():
        generated = model.generate(**tokens, max_new_tokens=4096, do_sample=False,
            use_cache=True, pad_token_id=tokenizer.eos_token_id)
    suffix = generated[0, tokens['input_ids'].shape[1]:]
    response = tokenizer.decode(suffix, skip_special_tokens=True)
    (output/'planner_response.txt').write_text(response+'\n', encoding='utf-8')
    probability, known, status = compile_response(response)
    save_json(output/'planner_status.json', status)
    if status['failed']:
        raise ValueError('LLM Plan 未通过结构校验，已保留原始响应；不会使用未知 Plan 伪装成功')
    obj, _ = canonical_parse(response)
    save_json(output/'plan.json', obj)
    np.savez(output/'plan.npz', events=probability, known=known, frames=obj['frames'])
    print(f"Plan 已通过校验：{obj['frames']} 帧，{len(obj['events'])} 个事件区间", flush=True)


def encode(config, output, device):
    """只编码原始用户文本；数值事件通过专用投影进入扩散网络。"""
    from transformers import AutoTokenizer, AutoModel
    from .text_embedding import encode_texts, DEFAULT_INSTRUCTION
    model_path = Path(config['models'])/'text_embedding'
    dtype = language_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', local_files_only=True)
    model = AutoModel.from_pretrained(model_path, torch_dtype=dtype,
        attn_implementation='sdpa', local_files_only=True).to(device).eval()
    model.config.use_cache = False
    query = f"Instruct: {DEFAULT_INSTRUCTION}\nQuery: {config['text']}"
    if len(tokenizer(query, truncation=False)['input_ids']) >= 512:
        raise ValueError('文本超出 embedding 预算，请缩短描述')
    vectors, _, truncated = encode_texts([config['text']], tokenizer, model, device, dtype, 512, DEFAULT_INSTRUCTION)
    if truncated or vectors.shape != (1, 2560) or not np.isfinite(vectors).all():
        raise ValueError('文本向量形状或数值异常')
    # 与训练缓存保持相同的 FP16 存储再转 FP32 约定。
    np.save(output/'text_embedding.npy', vectors.astype(np.float32))


def generate_motion(config, output, device):
    """加载 EMA 权重，在归一化空间中执行 DDIM，最后统一反归一化。"""
    from contextlib import nullcontext
    from .diffusion import SMDContinuousDiffusionConfig, ContinuousDiffusionConfig, SMDGaussianDiffusion
    from .events import EventDenoiser, pack_events
    from .motion_io import from_gravity_gauge, write_sps
    models = Path(config['models'])
    standing = config.get('boundary', 'none') == 'standing'
    boundary_model = config.get('boundary', 'none') in ('standing', 'free')
    checkpoint = Path(config.get('generator_checkpoint', models/('generator_boundary.pt' if boundary_model else 'generator.pt')))
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['format'] != ('boundary_events_v1' if boundary_model else 'sparse_events_v1'):
        raise ValueError('生成器权重格式不匹配')
    from .boundary import BoundaryEventDenoiser, sample_with_boundaries
    model = (BoundaryEventDenoiser if boundary_model else EventDenoiser)(SMDContinuousDiffusionConfig(**state['model_config']))
    model.load_state_dict(state['ema'], strict=True)
    del state['ema']
    model = model.to(device).eval()
    diff = SMDGaussianDiffusion(ContinuousDiffusionConfig(
        steps=state['diffusion_config']['steps'], x0_clip=state['diffusion_config']['x0_clip'])).to(device)
    with np.load(config.get('generation_stats', models/'generation_stats.npz'), allow_pickle=False) as stats:
        mean, std = stats['mean'].astype(np.float32), stats['std'].astype(np.float32)
    if mean.shape != (40,) or std.shape != (40,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('归一化统计量无效')
    with np.load(output/'plan.npz', allow_pickle=False) as data:
        n, probability = int(data['frames']), data['events']
    e = np.load(output/'text_embedding.npy', allow_pickle=False)
    row = pack_events(dict(num_frames=n, condition_embeddings=torch.from_numpy(e)), probability)
    names = ('condition_embeddings', 'condition_mask', 'condition_spans', 'phase_ids')
    conditions = [row[k][None].to(device) for k in names]
    conditions.append(torch.ones(1, n, dtype=torch.bool, device=device))
    if standing:
        from .references import boundary_condition, sha256
        width = config.get('boundary_frames',8)
        if n < 2*width+4:
            raise ValueError('预测动作太短，首尾约束后不足一个动作 patch；请减小 --boundary-frames 或增加文本中的动作时长')
        if not config.get('boundary_reference_path'):
            raise ValueError('Missing boundary reference snapshot; create a new request with run.py to use the bundled idle SPS')
        reference_path = Path(config['boundary_reference_path'])
        if config.get('boundary_reference_sha256') and sha256(reference_path) != config['boundary_reference_sha256']:
            raise ValueError('Boundary reference changed after request creation')
        pose_ref = np.load(reference_path,allow_pickle=False)
        if pose_ref.shape != (2,35) or not np.isfinite(pose_ref).all():
            raise ValueError('站姿参考文件无效')
        reference, known = boundary_condition(pose_ref, n, width, mean, std)
        reference = torch.from_numpy(reference).to(device)
        known = torch.from_numpy(known).to(device)
        batch = dict(zip((*names,'frame_mask'),conditions))
        np.save(output/'boundary_frame_mask.npy',known[0,:,0].cpu().numpy())
    elif boundary_model:
        # Disable both the pose-conditioning branch and every projection without
        # changing weights, semantic conditions, initial noise or DDIM updates.
        reference = torch.zeros(1, n, 40, device=device)
        known = torch.zeros_like(reference, dtype=torch.bool)
        batch = dict(zip((*names, 'frame_mask'), conditions))
        np.save(output/'boundary_frame_mask.npy', np.zeros(n, dtype=bool))
    # 显式独立随机数发生器，使模型初始化和语言推理不改变动作噪声。
    rng = torch.Generator(device=device).manual_seed(config['seed'])
    context = torch.autocast('cuda', dtype=torch.bfloat16) if config['precision'] == 'bf16' else nullcontext()
    with torch.inference_mode(), context:
        if boundary_model:
            x = sample_with_boundaries(model,diff,batch,reference,known,steps=config['steps'],
                cfg=config['cfg'],generator=rng,precision=config['precision'])
        else:
            x = diff.ddim_sample(model, (1, n, 40), *conditions,
                sampling_steps=config['steps'], guidance_scale=config['cfg'], generator=rng)
    motion = x[0].float().cpu().numpy()*std+mean
    if not np.isfinite(motion).all():
        raise ValueError('生成器输出存在 NaN 或 Inf')
    pose = from_gravity_gauge(motion)
    np.save(output/'motion40_raw.npy', motion)
    np.save(output/'pose35_raw.npy', pose)
    write_sps(output/'motion_raw.sps', pose, fps=20)


def postprocess(config, output, device):
    """平滑解码轨迹，同时导出便于工程使用的米、弧度、四元数数组。"""
    from scipy.spatial.transform import Rotation
    from .postprocess import smooth_pose
    from .motion_io import write_sps, SPS_JOINTS
    pose = np.load(output/'pose35_raw.npy', allow_pickle=False)
    result, report = smooth_pose(pose, config['root_hz'], config['joint_hz'], config['filter_order'])
    if config.get('boundary','none') == 'standing':
        # 在自由区间内平滑，端点作为锚点；整段已知站姿原样保留，避免滤波松开约束。
        w=config.get('boundary_frames',8)
        result=pose.copy()
        middle,report=smooth_pose(pose[w-1:len(pose)-w+1],config['root_hz'],config['joint_hz'],config['filter_order'])
        result[w-1:len(pose)-w+1]=middle
        mask=np.load(output/'boundary_frame_mask.npy',allow_pickle=False)
        assert np.array_equal(result[mask],pose[mask])
        report.update(boundary='standing',preserved_frames_each_end=w,
                      report_region='自由区间及两端锚点',boundary_max_error=0.)
    write_sps(output/'motion_smoothed.sps', result, fps=20)
    np.save(output/'pose35_smoothed.npy', result)
    qpos = np.concatenate([result[:, [2, 0, 1]]*.01,
        Rotation.from_euler('xyz', result[:, 3:6], degrees=True).as_quat(scalar_first=True),
        np.deg2rad(result[:, 6:])], axis=1)
    np.savez(output/'motion_smoothed_qpos.npz', qpos=qpos, fps=20,
        joint_names=np.array(SPS_JOINTS[1:]))
    save_json(output/'smoothing_report.json', report)


def contact_refine(config, output, device):
    """平滑后整段联合修复，更新接触标签，输出独立最终文件以保留中间结果。"""
    from .contact_window import refine_contact_window
    from .foot_labels import plan_support
    from .motion_io import write_sps, SPS_JOINTS
    from scipy.spatial.transform import Rotation
    pose=np.load(output/'pose35_smoothed.npy',allow_pickle=False)
    fixed=np.load(output/'boundary_frame_mask.npy') if config.get('boundary')=='standing' else np.zeros(len(pose),bool)
    mode=config.get('contact_refine','none')
    if mode not in ('heel_toe','joint','none'):
        raise ValueError('未知接触修复模式：'+str(mode))
    if mode in ('joint','heel_toe'):
        plan,_=plan_support(output/'plan.npz',len(pose))
        if mode=='heel_toe':
            from .contact_multipoint import refine_contact_multipoint
            result,history,report=refine_contact_multipoint(pose,plan,split_labels=True,fixed_frames=fixed)
            report.update(label_shape='T,foot,region',foot_order=['left','right'],region_order=['heel','toe'])
        else:
            result,history,report=refine_contact_window(pose,plan,update_labels=True,fixed_frames=fixed)
            report.update(label_shape='T,foot',foot_order=['left','right'])
        np.save(output/'contact_label_history.npy',history)
        np.save(output/'contact_labels_final.npy',history[-1])
        report.update(enabled=True,mode=mode,backend='CPU scipy sparse least_squares',fixed_boundary_max_error=float(np.abs(result[fixed]-pose[fixed]).max()) if fixed.any() else 0.)
    else:
        result=pose.copy();report=dict(enabled=False)
    if not np.isfinite(result).all():raise ValueError('接触修复输出包含非有限值')
    np.testing.assert_array_equal(result[fixed],pose[fixed])
    np.save(output/'pose35_final.npy',result)
    write_sps(output/'motion_final.sps',result,fps=20)
    qpos=np.concatenate([result[:,[2,0,1]]*.01,Rotation.from_euler('xyz',result[:,3:6],degrees=True).as_quat(scalar_first=True),np.deg2rad(result[:,6:])],axis=1)
    np.savez(output/'motion_final_qpos.npz',qpos=qpos,fps=20,joint_names=np.array(SPS_JOINTS[1:]))
    save_json(output/'contact_refine_report.json',report)


def run_stage(stage, config, output):
    torch.set_num_threads(config['cpu_threads'])
    device = torch.device(config['device'])
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    # 扩散核沿用验证时的数学注意力，避免换 kernel 带来的数值变化。
    if stage == 'diffusion':
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(False)
    start = time.perf_counter()
    {'planner': predict_plan, 'embedding': encode, 'diffusion': generate_motion,
        'smooth': postprocess, 'contact': contact_refine}[stage](config, output, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    report = dict(stage=stage, seconds=time.perf_counter()-start,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None)
    save_json(output/(stage+'_timing.json'), report)
    print(f"完成 {stage}：{report['seconds']:.1f} 秒", flush=True)
