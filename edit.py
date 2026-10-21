#!/usr/bin/env python3
"""Edit a G1 motion using the trained source-relative hand-height editor."""
from pathlib import Path
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EDIT_INSTRUCTION = (
    "Encode an instruction for editing a humanoid motion. Preserve the requested "
    "change, body side, direction, timing, speed, repetitions, and parts to keep unchanged."
)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def encode(config, output):
    """Load only the text encoder in this process; use the editor's training prompt."""
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from trace_core.text_embedding import encode_texts
    device = torch.device(config["device"])
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    location = Path(config["models"]) / "text_embedding"
    tokenizer = AutoTokenizer.from_pretrained(location, padding_side="left", local_files_only=True)
    model = AutoModel.from_pretrained(location, torch_dtype=dtype,
        attn_implementation="sdpa", local_files_only=True).to(device).eval()
    model.config.use_cache = False
    query = f"Instruct: {EDIT_INSTRUCTION}\nQuery: {config['instruction']}"
    if len(tokenizer(query, truncation=False)["input_ids"]) >= 512:
        raise ValueError("Edit instruction exceeds the embedding token budget")
    vectors, _, truncated = encode_texts([config["instruction"]], tokenizer, model,
        device, dtype, 512, EDIT_INSTRUCTION)
    if truncated or vectors.shape != (1, 2560) or not np.isfinite(vectors).all():
        raise ValueError("Invalid instruction embedding")
    np.save(output / "edit_embedding.npy", vectors.astype(np.float32))


def parse_llm(config, output):
    """Optional trained scope adapter; never replace it with the generation adapter."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from trace_core.edit_prompt import messages
    from trace_core.edit_requests import resolve_typed_time
    device = torch.device(config["device"])
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    base = Path(config["models"]) / "planner_base"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype,
        attn_implementation="sdpa", local_files_only=True).to(device)
    model = PeftModel.from_pretrained(model, config["scope_adapter"], local_files_only=True).eval()
    row = json.loads((output / "source_summary.json").read_text())
    prompt = tokenizer.apply_chat_template(messages(row, "source_summary"),
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    if batch["input_ids"].shape[1] + 384 > 4096:
        raise ValueError("Edit parser input exceeds its context budget")
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=384, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    (output / "parser_response.txt").write_text(response + "\n", encoding="utf-8")
    delta, _, _ = resolve_typed_time(json.loads(response), config["instruction"], config["frames"])
    save(output / "delta.json", delta)


def infer(config, output):
    import numpy as np
    import torch
    from trace_core.editor import SourceEventExtractor, revise_events
    from trace_core.edit_requests import compile_request
    from trace_core.edit_sampling import load_editor, load_stats, sample_edit
    from trace_core.motion_io import to_gravity_gauge, from_gravity_gauge, write_sps
    from trace_core.editor_events import PalmFK
    from trace_core.references import sha256
    torch.set_num_threads(4)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    device = config["device"]
    pose = np.load(output / "source_pose35.npy", allow_pickle=False)
    frames = len(pose)
    source = np.zeros((1, 104, 40), np.float32)
    source[0, :frames] = to_gravity_gauge(pose)
    lengths = np.array([frames])
    delta = json.loads((output / "delta.json").read_text())
    scope, request = compile_request(delta, frames)
    extractor = SourceEventExtractor(device, xml=ROOT / "assets/unitree_g1/g1_mocap_29dof.xml")
    events, confidence, palms, ground = extractor(source, lengths)
    revised, _ = revise_events(source, lengths, events, confidence, palms, ground, request[None])
    mean, std = load_stats(config["editing_stats"])
    model = load_editor(config["editor_checkpoint"], device)
    embedding = np.load(output / "edit_embedding.npy", allow_pickle=False)
    result = sample_edit(model, source, lengths, embedding, scope[None], request[None],
        revised, mean, std, ["interactive_edit"], device, config["precision"],
        config["steps"], config["cfg"], config["seed"])
    valid = np.arange(104) < frames
    protected = ~scope & valid[:, None]
    assert np.array_equal(result[0][protected], source[0][protected])
    # Reuse the supplied source's global root and protected joints exactly.
    # Gravity-gauge decoding otherwise resets the initial world XY and yaw.
    decoded = from_gravity_gauge(result[0, :frames])
    final = pose.copy()
    joint_scope = scope[:frames, 11:]
    final[:, 6:] = np.where(joint_scope, decoded[:, 6:], pose[:, 6:])
    np.save(output / "edited_motion40.npy", result[0, :frames], allow_pickle=False)
    np.save(output / "edited_pose35.npy", final, allow_pickle=False)
    np.save(output / "edit_mask.npy", scope[:frames], allow_pickle=False)
    np.save(output / "requested_offsets_m.npy", request[:frames], allow_pickle=False)
    np.save(output / "revised_events.npy", revised[0], allow_pickle=False)
    write_sps(output / "motion_edited.sps", final, fps=20)
    write_sps(output / "camera_reference.sps", np.concatenate((pose, final)), fps=20)
    with torch.inference_mode():
        tensor = torch.as_tensor(result, device=device)
        mask = torch.arange(104, device=device)[None] < frames
        updated_palms = PalmFK(ROOT / "assets/unitree_g1/g1_mocap_29dof.xml").to(device)(tensor, mask)[0].cpu().numpy()
    side = 0 if delta["effector"] == "left_hand" else 1
    active = np.abs(request[:frames, side]) > 1e-5
    achieved = updated_palms[:frames, side, 2] - palms[0, :frames, side]
    metrics = {
        "protected_encoded_coordinates_exact": True,
        "protected_pose_coordinates_exact": bool(np.array_equal(
            final[:, 6:][~joint_scope], pose[:, 6:][~joint_scope])),
        "root_pose_exact": bool(np.array_equal(final[:, :6], pose[:, :6])),
        "palm_relative_height_mae_cm": float(np.abs(achieved[active] - request[:frames, side][active]).mean() * 100),
        "mean_requested_palm_offset_cm": float(request[:frames, side][active].mean() * 100),
        "mean_achieved_palm_offset_cm": float(achieved[active].mean() * 100),
        "note": "Palm-height diagnostic; not the paper's wrist-based success metric.",
    }
    save(output / "result.json", dict(success=True, delta=delta, frames=frames, fps=20,
        parser=config["parser"], metrics=metrics, postprocessing=False, ik=False,
        checkpoint_sha256=sha256(config["editor_checkpoint"]),
        source_sha256=config["source_sha256"], steps=config["steps"], cfg=config["cfg"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="20 FPS G1 SPS or pose35 NPY")
    parser.add_argument("--instruction", help="Explicit source-relative hand-height edit")
    parser.add_argument("--delta", type=Path, help="Validated typed edit JSON (optional)")
    parser.add_argument("--scope-adapter", type=Path, help="Optional trained LLM edit-scope adapter")
    parser.add_argument("--parser", choices=("llm", "explicit"), default="llm",
                        help="Trained LLM parser, or a restricted explicit template parser")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--editor-checkpoint", type=Path)
    parser.add_argument("--editing-stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-device", type=int, default=0)
    parser.add_argument("--stage", choices=("encode", "infer", "parse"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.stage:
        config = json.loads((output / "request.json").read_text())
        {"encode": encode, "infer": infer, "parse": parse_llm}[args.stage](config, output)
        return
    if args.source is None or not args.instruction or not args.instruction.strip():
        parser.error("--source and --instruction are required")
    if not 1 <= args.steps <= 300 or not 0 <= args.cfg <= 20:
        parser.error("Editing requires 1..300 steps and CFG in [0,20]")
    if args.delta and args.scope_adapter:
        parser.error("Choose a typed delta or a trained scope adapter")
    import numpy as np
    from trace_core.references import read_poses, sha256
    from trace_core.edit_requests import compile_request, explicit_training_request
    pose = read_poses(args.source)
    if not 2 <= len(pose) <= 104:
        parser.error("The trained editor supports 2..104 frames; no cropping or rescaling is applied")
    if args.source.suffix.lower() == ".sps":
        match = re.search(r"^# input_fps:\s*(\d+(?:\.\d+)?)", args.source.read_text(), re.M)
        if not match or float(match[1]) != 20:
            parser.error("Editing expects 20 FPS source motion; resample explicitly before editing")
    models = args.models.resolve()
    if not args.delta and args.parser == "llm":
        args.scope_adapter = args.scope_adapter or models / "edit_scope_adapter"
    if args.parser == "explicit" and args.scope_adapter:
        parser.error("--scope-adapter requires --parser llm")
    checkpoint = (args.editor_checkpoint or models / "editor.pt").resolve()
    stats = (args.editing_stats or models / "editing_stats.npz").resolve()
    for path in (checkpoint, stats, models / "text_embedding/config.json"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.scope_adapter and not (args.scope_adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("A separate trained edit-scope adapter is required")
    if args.delta:
        delta = json.loads(args.delta.read_text())
        compile_request(delta, len(pose))
        mode = "typed_delta"
    elif args.scope_adapter:
        delta, mode = None, "trained_llm_scope_adapter"
    else:
        delta, _, _ = explicit_training_request(args.instruction, len(pose))
        mode = "explicit_template_parser"
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "source_pose35.npy", pose, allow_pickle=False)
    from trace_core.motion_io import write_sps
    write_sps(output / "source.sps", pose, fps=20)
    config = dict(instruction=args.instruction.strip(), frames=len(pose), parser=mode,
        models=str(models), editor_checkpoint=str(checkpoint), editing_stats=str(stats),
        device=args.device, precision=args.precision, steps=args.steps, cfg=args.cfg,
        seed=args.seed, source_sha256=sha256(args.source),
        scope_adapter=str(args.scope_adapter.resolve()) if args.scope_adapter else None)
    save(output / "request.json", config)
    if delta is not None:
        save(output / "delta.json", delta)
    else:
        import torch
        from trace_core.editor_events import PalmFK
        from trace_core.motion_io import to_gravity_gauge
        source = torch.from_numpy(to_gravity_gauge(pose))[None]
        with torch.inference_mode():
            palms = PalmFK(ROOT / "assets/unitree_g1/g1_mocap_29dof.xml")(
                source, torch.ones((1, len(pose)), dtype=torch.bool))[0, ..., 2].numpy()
        heights = [[i, *np.round(palms[i:i+4].mean(0) * 100, 2).tolist()]
                   for i in range(0, len(pose), 4)]
        save(output / "source_summary.json", dict(frames=len(pose), fps=20,
            instruction=args.instruction, source_hand_heights_cm=heights))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    try:
        for stage in (["parse"] if args.scope_adapter else []) + ["encode", "infer"]:
            subprocess.run([sys.executable, str(ROOT / "edit.py"), "--stage", stage,
                            "--output", str(output)], env=env, check=True)
        if not args.skip_render:
            for name, source in (("before", output / "source.sps"),
                                 ("after", output / "motion_edited.sps")):
                subprocess.run([sys.executable, str(ROOT / "render.py"),
                    "--input", str(source), "--output", str(output / name),
                    "--prompt", args.instruction, "--render-device", str(args.render_device),
                    "--camera-reference", str(output / "camera_reference.sps")],
                    env=env, check=True)
    except Exception as exc:
        save(output / "FAILED.json", {"error": str(exc)})
        raise
    print(f"Edited motion: {output / 'motion_edited.sps'}", flush=True)


if __name__ == "__main__":
    main()
