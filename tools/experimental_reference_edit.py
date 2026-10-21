"""Exploratory source-noised resampling with the boundary generator.

This is NOT the separately trained relative-hand editor. A revised full caption
and its LLM Plan guide masked denoising from a noised source motion. An optional
jump target adds differentiable FK guidance inside sampling. No post-smoothing
or external IK is applied.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def make_mask(frames, body, boundary):
    import numpy as np
    if not 1 <= boundary <= 16 or frames < 2*boundary+4:
        raise ValueError("The requested idle bands leave too few editable frames")
    mask = np.zeros((frames, 40), bool)
    groups = {
        "lower_body": [(2, 3), (5, 26)],
        "right_leg": [(2, 3), (5, 11), (17, 26)],
        "left_leg": [(2, 3), (5, 17), (23, 26)],
        "torso": [(5, 11), (23, 26)],
        "whole_body": [(2, 3), (5, 40)],
    }
    for lo, hi in groups[body]:
        mask[boundary:frames-boundary, lo:hi] = True
    return mask


def resample(model, diffusion, batch, source, editable, strength, steps, cfg, seed,
             guidance=None):
    """Start from a noised source and restore protected states at every step."""
    import torch
    known = ~editable
    start = max(1, round(strength * (diffusion.steps-1)))
    indices = torch.linspace(start, 0, min(steps, start+1), device=source.device).round().long().unique_consecutive()
    noise = torch.randn(source.shape, device=source.device,
                        generator=torch.Generator(device=source.device).manual_seed(seed))
    x = diffusion.alpha_bar[start].sqrt()*source + (1-diffusion.alpha_bar[start]).sqrt()*noise
    args = [batch[name] for name in
            ("condition_embeddings", "condition_mask", "condition_spans", "phase_ids", "frame_mask")]
    with torch.inference_mode():
        for i, t in enumerate(indices):
            ts = torch.full((1,), int(t), device=source.device, dtype=torch.long)
            kw = dict(boundary_reference=source, boundary_mask=known)
            conditional = model(x, ts, *args, **kw)
            unconditional = model(x, ts, *args, force_unconditional=True, **kw)
            clean = (unconditional.float()+cfg*(conditional.float()-unconditional.float())).clamp(
                -diffusion.config.x0_clip, diffusion.config.x0_clip)
            clean = torch.where(known, source, clean)
            if guidance is not None:
                clean = guidance(clean, (i+1)/len(indices), final=i+1 == len(indices))
                clean = torch.where(known, source, clean)
            if i+1 == len(indices):
                x = clean
            else:
                alpha, nxt = diffusion.alpha_bar[t], diffusion.alpha_bar[indices[i+1]]
                residual = (x-alpha.sqrt()*clean)/(1-alpha).sqrt().clamp_min(1e-8)
                x = nxt.sqrt()*clean+(1-nxt).sqrt()*residual
                reference = nxt.sqrt()*source+(1-nxt).sqrt()*noise
                x = torch.where(known, reference, x)
    return torch.where(known, source, x)


def infer(config, output):
    import numpy as np
    import torch
    from trace_core.references import read_poses
    from trace_core.motion_io import (to_gravity_gauge, from_gravity_gauge, write_sps,
        euler_xyz_to_matrix, matrix_to_euler_xyz, heading_basis_from_yaw, yaw_from_rotation)
    from trace_core.diffusion import SMDContinuousDiffusionConfig, ContinuousDiffusionConfig, SMDGaussianDiffusion
    from trace_core.boundary import BoundaryEventDenoiser
    from trace_core.events import pack_events
    from trace_core.edit_sampling import load_stats
    torch.set_num_threads(4)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    device = torch.device(config["device"])
    pose = read_poses(output / "source_pose35.npy")
    n = len(pose)
    raw = to_gravity_gauge(pose)
    mask = make_mask(n, config["body"], config["boundary_frames"])
    if config.get("active_span") is not None:
        start, end = config["active_span"]
        if not config["boundary_frames"] <= start < end <= n-config["boundary_frames"]:
            raise ValueError("Edit span must be inside the unprotected source interval")
        mask[:start] = False
        mask[end:] = False
    if config.get("height_gain_cm") is not None or config.get("pelvis_delta_cm") is not None:
        # A vertical jump edit needs root height and both legs, not a different
        # upper-body celebration. Preserve the source tilt, waist and arms.
        mask[:, 5:11] = False
        mask[:, 23:] = False
    if config.get('relative_attribute') is not None:
        from trace_core.relative_attribute_guidance import relative_attribute_mask
        mask = relative_attribute_mask(n, config['active_span'], config['relative_attribute'],
                                       1 if config['body'] == 'right_leg' else 0)
    mean, std = load_stats(Path(config.get("generation_stats", Path(config["models"]) / "generation_stats.npz")))
    state = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
    if state["format"] != "boundary_events_v1":
        raise ValueError("This experimental path requires a boundary-conditioned generator")
    model = BoundaryEventDenoiser(SMDContinuousDiffusionConfig(**state["model_config"]))
    model.load_state_dict(state["ema"], strict=True)
    model = model.to(device).eval()
    diff = SMDGaussianDiffusion(ContinuousDiffusionConfig(
        steps=state["diffusion_config"]["steps"],
        x0_clip=state["diffusion_config"]["x0_clip"])).to(device)
    with np.load(output / "plan.npz") as plan:
        old_n, old_events = int(plan["frames"]), plan["events"]
    # Keep source duration; align the new Plan in normalized time, never warp source motion.
    count, old_count = (n+3)//4, (old_n+3)//4
    idx = np.minimum(((np.arange(count)+.5)/count*old_count).astype(int), old_count-1)
    events = np.zeros((128, 16), np.float32)
    events[:count] = old_events[idx]
    embedding = torch.from_numpy(np.load(output / "text_embedding.npy"))
    row = pack_events(dict(num_frames=n, condition_embeddings=embedding), events)
    batch = {key: row[key][None].to(device) for key in
             ("condition_embeddings", "condition_mask", "condition_spans", "phase_ids")}
    batch["frame_mask"] = torch.ones(1, n, dtype=torch.bool, device=device)
    source = torch.from_numpy((raw-mean)/std)[None].to(device)
    editable = torch.from_numpy(mask)[None].to(device)
    guidance = None
    if config.get("height_gain_cm") is not None:
        from trace_core.jump_guidance import JumpHeightGuidance
        if config.get("active_span") is None:
            raise ValueError("Jump targets require an explicit active interval")
        guidance = JumpHeightGuidance(source, mean, std, editable,
                                      config["active_span"], config["height_gain_cm"])
    if config.get("pelvis_delta_cm") is not None:
        from trace_core.crouch_guidance import CrouchDepthGuidance
        if config.get("height_gain_cm") is not None or config.get("active_span") is None:
            raise ValueError("Crouch guidance needs one target and an explicit active interval")
        guidance = CrouchDepthGuidance(source, mean, std, editable,
                                       config["active_span"], config["pelvis_delta_cm"])
        np.save(output / "target_root_height_cm.npy", guidance.target_cm.cpu().numpy())
    if config.get('relative_attribute') is not None:
        from trace_core.relative_attribute_guidance import RelativeAttributeGuidance
        guidance = RelativeAttributeGuidance(source, mean, std, editable, config['active_span'],
            config['relative_attribute'], 1 if config['body'] == 'right_leg' else 0, config['attribute_delta'])
        np.save(output/'target_attribute.npy', guidance.target.cpu().numpy())
    sampled = resample(model, diff, batch, source, editable, config["strength"],
                       config["steps"], config["cfg"], config["seed"], guidance=guidance)
    if config.get("pelvis_delta_cm") is not None:
        from trace_core.crouch_guidance import validate_crouch_report
        validate_crouch_report(guidance.report)
    if config.get('relative_attribute') is not None:
        from trace_core.relative_attribute_guidance import validate_attribute_report
        validate_attribute_report(guidance.report)
    result = np.where(mask, sampled[0].cpu().numpy()*std+mean, raw).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite motion output")
    decoded = from_gravity_gauge(result)
    final = pose.copy()
    final[:, 6:] = np.where(mask[:, 11:], decoded[:, 6:], pose[:, 6:])
    final[:, 1] = np.where(mask[:, 2], decoded[:, 1], pose[:, 1])
    initial_yaw = yaw_from_rotation(euler_xyz_to_matrix(pose[0, 3:6]))
    rotations = heading_basis_from_yaw(initial_yaw) @ euler_xyz_to_matrix(decoded[:, 3:6])
    final[:, 3:6] = np.where(mask[:, 5:6], matrix_to_euler_xyz(rotations), pose[:, 3:6])
    w = config["boundary_frames"]
    assert np.array_equal(final[:w], pose[:w]) and np.array_equal(final[-w:], pose[-w:])
    assert np.array_equal(result[~mask], raw[~mask])
    np.save(output / "edited_pose35.npy", final)
    np.save(output / "edited_motion40.npy", result)
    np.save(output / "edit_mask.npy", mask)
    write_sps(output / "motion_edited.sps", final, fps=20)
    write_sps(output / "camera_reference.sps", np.concatenate((pose, final)), fps=20)
    save(output / "result.json", dict(success=True, frames=n, fps=20,
        method="experimental_generator_reference_resampling",
        source_frames=n, predicted_target_frames=old_n,
        target_plan_alignment="normalized-time nearest-patch mapping; source motion not resampled",
        body=config["body"], strength=config["strength"], postprocessing=False, ik=False,
        active_span=config.get("active_span"),
        numerical_guidance=guidance.report if guidance else None,
        in_sampling_kinematic_optimization=guidance is not None,
        protected_encoded_coordinates_exact=True, idle_bands_exact=True,
        note="Pipeline completion, not an assessment of edit success. No new editor training."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--text", help="Revised full action caption")
    parser.add_argument("--instruction", help="Follow-up instruction for the comparison caption")
    parser.add_argument("--body", choices=["lower_body", "right_leg", "left_leg", "torso", "whole_body"])
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--generator-checkpoint", type=Path)
    parser.add_argument("--generation-stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=.45)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=2.5)
    parser.add_argument("--boundary-frames", type=int, default=8)
    parser.add_argument("--active-span", type=int, nargs=2, metavar=("START", "END"),
                        help="Optional half-open source-frame interval for local editing")
    parser.add_argument("--height-gain-cm", type=float,
                        help="Experimental numerical jump-height guidance during denoising")
    parser.add_argument("--pelvis-delta-cm", type=float,
                        help="Source-relative crouch height change: negative deeper, positive shallower")
    parser.add_argument('--relative-attribute', choices=['kick_height', 'knee_height', 'bow_angle'])
    parser.add_argument('--attribute-delta', type=float, help='Signed relative cm, or degrees for bow_angle')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-device", type=int, default=0)
    parser.add_argument("--skip-render", action="store_true",
                        help="Keep motion outputs for a separate overlay renderer")
    parser.add_argument("--stage", choices=["planner", "embedding", "infer"])
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage:
        config = json.loads((output / "request.json").read_text())
        if args.stage == "infer":
            infer(config, output)
        else:
            from trace_core.pipeline import run_stage
            run_stage(args.stage, config, output)
        return
    if not all((args.source, args.text, args.body, args.instruction)):
        parser.error("--source, --text, --instruction and --body are required")
    if args.pelvis_delta_cm is not None and (args.height_gain_cm is not None or args.body != "lower_body"):
        parser.error("--pelvis-delta-cm requires lower_body and cannot be combined with jump guidance")
    if args.relative_attribute is not None:
        body_ok = args.body == 'torso' if args.relative_attribute == 'bow_angle' else args.body in {'left_leg', 'right_leg'}
        if not body_ok or args.attribute_delta is None or args.active_span is None or args.height_gain_cm is not None or args.pelvis_delta_cm is not None:
            parser.error('Relative attributes need a matching body, signed delta, explicit span and no other numerical target')
    elif args.attribute_delta is not None:
        parser.error('--attribute-delta requires --relative-attribute')
    if not .01 <= args.strength <= .8 or not 1 <= args.steps <= 1000:
        parser.error("Strength must be in [0.01,0.8] and steps in [1,1000]")
    import numpy as np
    from trace_core.references import read_poses, sha256
    from trace_core.motion_io import write_sps
    if args.source.suffix.lower() == ".sps":
        import re
        fps = re.search(r"^# input_fps:\s*(\d+(?:\.\d+)?)", args.source.read_text(), re.M)
        if not fps or float(fps[1]) != 20:
            parser.error("Source SPS must be explicitly tagged as 20 FPS")
    pose = read_poses(args.source)
    if not 20 <= len(pose) <= 512:
        parser.error("Source must have 20..512 frames")
    make_mask(len(pose), args.body, args.boundary_frames)
    models = args.models.resolve()
    checkpoint = (args.generator_checkpoint or models / "generator_boundary.pt").resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "source_pose35.npy", pose)
    write_sps(output / "source.sps", pose, fps=20)
    config = dict(text=args.text, instruction=args.instruction, body=args.body,
        models=str(models), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        source_sha256=sha256(args.source), strength=args.strength, steps=args.steps,
        cfg=args.cfg, seed=args.seed, boundary_frames=args.boundary_frames, device=args.device,
        cpu_threads=4, experimental=True, active_span=args.active_span,
        height_gain_cm=args.height_gain_cm,
        pelvis_delta_cm=args.pelvis_delta_cm,
        relative_attribute=args.relative_attribute, attribute_delta=args.attribute_delta,
        generation_stats=str((args.generation_stats or models / "generation_stats.npz").resolve()))
    save(output / "request.json", config)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    try:
        for stage in ("planner", "embedding", "infer"):
            subprocess.run([sys.executable, __file__, "--stage", stage, "--output", str(output)],
                           env=env, check=True)
        if args.skip_render:
            return
        for name, src in (("before", output/"source.sps"), ("after", output/"motion_edited.sps")):
            subprocess.run([sys.executable, str(ROOT/"render.py"), "--input", str(src),
                "--output", str(output/name), "--prompt", args.instruction,
                "--render-device", str(args.render_device),
                "--camera-reference", str(output/"camera_reference.sps")], env=env, check=True)
    except Exception as exc:
        save(output/"FAILED.json", dict(error=str(exc)))
        raise


if __name__ == "__main__":
    main()
