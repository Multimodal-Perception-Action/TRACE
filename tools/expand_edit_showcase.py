"""Add actual editing trials to a previous showcase while retaining every result."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_edit_showcase import execute, save

SOURCES = [
    ("wave_right", "Briefly wave with the right hand."),
    ("punch_right", "Briefly punch forward with the right arm."),
    ("knee_left", "Briefly lift the left knee, then return to standing."),
    ("side_lean", "Briefly lean the upper body to the left, then stand upright."),
    ("lunge", "Perform a short forward lunge with the right leg, then return to standing."),
    ("tiptoe", "Briefly rise onto both toes, then return to standing."),
]
CASES = [
    dict(id="E11", source="wave_right", method="trained", title="Wave: lower the hand",
         instruction="Lower the right hand by 8 cm during the middle half."),
    dict(id="E12", source="wave_right", method="trained", title="Wave: raise the hand",
         instruction="Raise the right hand by 6 cm during the middle half."),
    dict(id="E13", source="punch_right", method="trained", title="Punch at a lower height",
         instruction="Lower the right hand by 6 cm during the middle half."),
    dict(id="E14", source="punch_right", method="trained", title="Punch: a brief higher gesture",
         instruction="Raise the right hand by 6 cm during the middle third."),
    dict(id="E15", source="arms_out", method="trained", title="Arms spread: raise the left hand",
         instruction="Raise the left hand by 8 cm during the middle half."),
    dict(id="E16", source="arms_out", method="trained", title="Arms spread: raise the right hand",
         instruction="Raise the right hand by 8 cm during the middle half."),
    dict(id="E17", source="knee_left", method="experimental", title="Lift the left knee higher",
         instruction="Lift the left knee higher, then return to standing.",
         text="Briefly lift the left knee high toward the chest, then return to standing.",
         body="left_leg"),
    dict(id="E18", source="knee_left", method="experimental", title="Lift the left knee less",
         instruction="Keep the left knee lower with a smaller lift.",
         text="Briefly lift the left knee only a little, keeping the foot close to the ground, then stand.",
         body="left_leg"),
    dict(id="E19", source="lunge", method="experimental", title="A deeper forward lunge",
         instruction="Lower the pelvis and bend the knees more during the lunge.",
         text="Perform a deep forward lunge with the right leg, lowering the pelvis, then return to standing.",
         body="lower_body"),
    dict(id="E20", source="lunge", method="experimental", title="A shallower forward lunge",
         instruction="Bend the knees less and keep the lunge shallow.",
         text="Perform a shallow forward lunge with the right leg and a small knee bend, then return to standing.",
         body="lower_body"),
    dict(id="E21", source="side_lean", method="experimental", title="Lean to the other side",
         instruction="Lean the upper body to the right instead of the left.",
         text="Briefly lean the upper body to the right, then stand upright.",
         body="torso"),
    dict(id="E22", source="side_lean", method="experimental", title="A smaller sideways lean",
         instruction="Make the leftward lean smaller and stay closer to upright.",
         text="Briefly make a very slight upper-body lean to the left, then stand upright.",
         body="torso"),
    dict(id="E23", source="tiptoe", method="experimental", title="Rise higher onto the toes",
         instruction="Raise the heels higher while rising onto both toes.",
         text="Briefly rise high onto both toes, raising both heels well off the floor, then return to standing.",
         body="lower_body"),
    dict(id="E24", source="bow", method="experimental", title="A shallower bow",
         instruction="Bow less deeply and keep the torso more upright.",
         text="Briefly perform a shallow forward bow, then return to standing.",
         body="torso"),
]


def prepare(previous, output):
    old = json.loads((previous / "manifest.json").read_text())
    expected = dict(old, sources=old["sources"] + [dict(id=k, text=v) for k, v in SOURCES],
                    cases=old["cases"] + CASES, previous_showcase=str(previous),
                    expanded_at=time.time())
    output.mkdir(parents=True, exist_ok=True)
    for folder in ("sources", "edits", "logs"):
        (output / folder).mkdir(exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        current = json.loads(manifest.read_text())
        if current["cases"] != expected["cases"] or current["sources"] != expected["sources"]:
            raise ValueError("Existing manifest has different cases")
    else:
        save(manifest, expected)
    for kind, records in (("sources", old["sources"]), ("edits", old["cases"])):
        for row in records:
            source = previous / kind / row["id"]
            destination = output / kind / row["id"]
            if not source.is_dir():
                raise FileNotFoundError(source)
            if destination.exists():
                if destination.resolve() != source.resolve():
                    raise ValueError(f"Unexpected existing path: {destination}")
            else:
                destination.symlink_to(source, target_is_directory=True)
    reference = output / "idle_pose35.npy"
    if reference.exists():
        if reference.read_bytes() != (previous / reference.name).read_bytes():
            raise ValueError("Idle reference mismatch")
    else:
        shutil.copy2(previous / reference.name, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--phase", choices=["generate", "edit", "all"], default="all")
    args = parser.parse_args()
    previous, output = args.previous.resolve(), args.output.resolve()
    if previous == output:
        parser.error("Use a new output folder")
    prepare(previous, output)
    import numpy as np
    from trace_core.references import read_poses

    def run_phase(phase):
        jobs = SOURCES if phase == "generate" else CASES
        def worker(index):
            gpu = args.gpus[index]
            results = []
            for job in jobs[index::len(args.gpus)]:
                name = job[0] if phase == "generate" else job["id"]
                directory = output / ("sources" if phase == "generate" else "edits") / name
                if (directory / "result.json").exists():
                    results.append(dict(id=name, status="existing"))
                    continue
                try:
                    if directory.exists():
                        raise FileExistsError(f"Incomplete output retained for inspection: {directory}")
                    if phase == "generate":
                        command = [sys.executable, str(ROOT / "run.py"), "--text", job[1],
                            "--start-ref", str(output / "idle_pose35.npy"),
                            "--end-ref", str(output / "idle_pose35.npy"),
                            "--contact-refine", "none", "--skip-render", "--output", str(directory)]
                    else:
                        source = output / "sources" / job["source"] / "motion_final.sps"
                        if job["method"] == "trained":
                            n = len(read_poses(source))
                            if not 32 <= n <= 104:
                                raise ValueError(f"Uncropped source length {n} is outside editor range")
                            command = [sys.executable, str(ROOT / "edit.py"),
                                "--source", str(source), "--instruction", job["instruction"],
                                "--skip-render", "--output", str(directory)]
                        else:
                            command = [sys.executable, str(ROOT / "tools/experimental_reference_edit.py"),
                                "--source", str(source), "--instruction", job["instruction"],
                                "--text", job["text"], "--body", job["body"],
                                "--skip-render", "--output", str(directory)]
                    print(f"{phase} {name} on GPU {gpu}", flush=True)
                    execute(command, directory, gpu, output / "logs" / f"{phase}_{name}.log")
                    if phase == "edit":
                        before = np.load(directory / "source_pose35.npy")
                        after = np.load(directory / "edited_pose35.npy")
                        if not (np.array_equal(before[:8], after[:8]) and
                                np.array_equal(before[-8:], after[-8:])):
                            raise AssertionError("Protected idle frames changed")
                    results.append(dict(id=name, status="complete"))
                    print(f"complete {name}", flush=True)
                except Exception as exc:
                    results.append(dict(id=name, status="failed", error=str(exc)))
                    print(f"failed {name}: {exc}", flush=True)
            return results
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            results = [row for group in pool.map(worker, range(len(args.gpus))) for row in group]
        save(output / f"{phase}_status.json", dict(jobs=results, time=time.time()))
        return not any(row["status"] == "failed" for row in results)

    for phase in (["generate", "edit"] if args.phase == "all" else [args.phase]):
        if not run_phase(phase):
            raise SystemExit("Some jobs failed; retained logs and outputs, without success filtering.")


if __name__ == "__main__":
    main()
