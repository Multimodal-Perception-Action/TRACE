#!/usr/bin/env python3
"""Edit the latest motion using a follow-up instruction and persistent context."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from trace_core.conversation import atomic_json, source_context, summarize_motion, resolve_with_llm

ROOT = Path(__file__).resolve().parent


def execute(command, environment):
    subprocess.run(list(map(str, command)), cwd=ROOT, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--instruction", help="Only the change requested in this turn")
    parser.add_argument("--source", type=Path, help="Initial generated directory or SPS; first turn only")
    parser.add_argument("--source-text", help="Original caption, needed once for imported motions without metadata")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--generator-checkpoint", type=Path, help="Compatible boundary generator; first turn only")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--render-device", type=int, default=0)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--stage", choices=["resolve"], help=argparse.SUPPRESS)
    parser.add_argument("--context", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage:
        resolve_with_llm(args.context, args.models / "planner_base", args.device)
        return
    if not args.instruction or not args.instruction.strip() or len(args.instruction) > 2000:
        parser.error("Provide a nonempty instruction of at most 2000 characters")
    session = args.session.resolve()
    state_path = session / "session.json"
    if not state_path.exists() and args.source is None:
        parser.error("A new session needs --source once")
    session.mkdir(parents=True, exist_ok=True)
    with (session / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This session already has an active editing turn")
        run_turn(args, session, state_path)


def run_turn(args, session, state_path):
    from trace_core.references import read_poses, sha256, validate_checkpoint
    from trace_core.motion_io import write_sps
    if state_path.exists():
        if args.source is not None or args.source_text is not None or args.generator_checkpoint is not None:
            raise ValueError("An existing session already defines its source and checkpoint")
        state = json.loads(state_path.read_text())
        if state.get("schema") != "trace_conversation_v1":
            raise ValueError("Unknown session schema")
    else:
        source, caption, request, poses = source_context(args.source, args.source_text)
        models = args.models.resolve()
        checkpoint = Path(args.generator_checkpoint or request.get("generator_checkpoint")
                          or request.get("checkpoint") or models / "generator_boundary.pt").resolve()
        validate_checkpoint(checkpoint, "standing")
        stats = Path(request.get("generation_stats") or models / "generation_stats.npz").resolve()
        initial = session / "initial.sps"
        if initial.exists():
            raise ValueError("Session initialization was interrupted; use a new session directory")
        write_sps(initial, poses, fps=20)
        state = dict(schema="trace_conversation_v1", initial_caption=caption, latest_caption=caption,
                     models=str(models), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                     generation_stats=str(stats), stats_sha256=sha256(stats), boundary_frames=8,
                     initial_source_sha256=sha256(source), current_motion="initial.sps",
                     current_sha256=sha256(initial), history=[])
        atomic_json(state_path, state)
    current = (session / state["current_motion"]).resolve()
    current.relative_to(session)
    if sha256(current) != state["current_sha256"]:
        raise ValueError("Current motion changed outside this session")
    for path, expected in ((state["checkpoint"], state["checkpoint_sha256"]),
                           (state["generation_stats"], state["stats_sha256"])):
        if sha256(Path(path)) != expected:
            raise ValueError("A session model/statistics artifact changed")
    turns = session / "turns"
    turns.mkdir(exist_ok=True)
    number = 1 + max((int(p.name) for p in turns.iterdir() if p.is_dir() and p.name.isdigit()), default=0)
    turn = turns / f"{number:03d}"
    turn.mkdir()
    summary = summarize_motion(read_poses(current), state["boundary_frames"])
    context = dict(initial_caption=state["initial_caption"], latest_caption=state["latest_caption"],
                   previous_requests=[dict(instruction=x["instruction"], target_caption=x["target_caption"],
                                           action=x["action"]) for x in state["history"][-6:]],
                   source_summary=summary, instruction=args.instruction.strip())
    atomic_json(turn / "context.json", context)
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    started = time.monotonic()
    try:
        execute([sys.executable, ROOT / "converse.py", "--session", session, "--stage", "resolve",
                 "--context", turn / "context.json", "--models", state["models"], "--device", args.device], environment)
        intent = json.loads((turn / "intent.json").read_text())
        resolved = json.loads((turn / "compiled_edit.json").read_text())
        print("Resolved request: " + json.dumps(resolved, ensure_ascii=False), flush=True)
        command = [sys.executable, ROOT / "tools/experimental_reference_edit.py",
                   "--source", current, "--text", resolved["text"], "--instruction", args.instruction,
                   "--body", resolved["body"], "--active-span", *resolved["active_span"],
                   "--strength", resolved["strength"], "--cfg", resolved["cfg"],
                   "--boundary-frames", state["boundary_frames"], "--models", state["models"],
                   "--generator-checkpoint", state["checkpoint"], "--generation-stats", state["generation_stats"],
                   "--device", args.device, "--skip-render", "--output", turn / "edit"]
        if resolved["height_gain_cm"] is not None:
            command += ["--height-gain-cm", resolved["height_gain_cm"]]
        if resolved.get("pelvis_delta_cm") is not None:
            command += ["--pelvis-delta-cm", resolved["pelvis_delta_cm"]]
        if resolved.get("relative_attribute") is not None:
            command += ["--relative-attribute", resolved["relative_attribute"], "--attribute-delta", resolved["attribute_delta"]]
        atomic_json(turn / "command.json", list(map(str, command)))
        execute(command, environment)
        after = turn / "edit/motion_edited.sps"
        after_summary = summarize_motion(read_poses(after), state["boundary_frames"])
        result = json.loads((turn / "edit/result.json").read_text())
        if not result.get("protected_encoded_coordinates_exact") or not result.get("idle_bands_exact"):
            raise ValueError("The edit did not pass reference-preservation checks")
        record = dict(turn=number, instruction=args.instruction.strip(), action=intent["action"],
                      target_caption=resolved["text"], source=str(current.relative_to(session)),
                      source_sha256=sha256(current), motion=str(after.relative_to(session)),
                      motion_sha256=sha256(after), source_summary=summary, output_summary=after_summary,
                      compiled_edit=resolved, seconds=time.monotonic()-started)
        atomic_json(turn / "completed.json", record)
        # Commit the new reference only after inference and preservation checks pass.
        state["current_motion"] = record["motion"]
        state["current_sha256"] = record["motion_sha256"]
        state["latest_caption"] = resolved["text"]
        state["history"].append(record)
        atomic_json(state_path, state)
    except Exception as error:
        atomic_json(turn / "FAILED.json", dict(error=str(error), reference_unchanged=True))
        raise
    if not args.skip_render:
        try:
            execute([sys.executable, ROOT / "tools/render_edit_overlay.py", "--edit", turn / "edit",
                     "--output", turn / "overlay", "--priority", "before", "--device", args.render_device], environment)
        except Exception as error:
            atomic_json(turn / "RENDER_FAILED.json", dict(error=str(error), motion_completed=True))
            raise
    print(f"Turn {number} complete: {after}", flush=True)
    print("Next turn: keep --session and supply only a new --instruction.", flush=True)


if __name__ == "__main__":
    main()
