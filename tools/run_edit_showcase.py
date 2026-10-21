"""Run genuine generation/edit examples with one fixed idle pose at both ends.

Outputs are local review material. Reference-resampling cases are explicitly
separated from the trained relative-hand editor.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = [
    ("arms_out", "Briefly spread both arms sideways."),
    ("hands_up", "Briefly raise both hands."),
    ("squat", "A person crouches down and stands up."),
    ("jump", "Perform a jumping jack."),
    ("kick", "A person kicks forward with the right leg, then returns to standing."),
    ("bow", "Briefly bow forward, then return to standing."),
]
CASES = [
    dict(id="E01", source="arms_out", method="trained", title="Lower the right hand",
         instruction="Lower the right hand by 8 cm during the middle half."),
    dict(id="E02", source="hands_up", method="trained", title="Lower the left hand",
         instruction="Lower the left hand by 8 cm during the middle half."),
    dict(id="E03", source="E02", method="trained", title="Second turn: lower the other hand",
         instruction="Lower the right hand by 8 cm during the middle half."),
    dict(id="E04", source="hands_up", method="trained", title="A shorter edit interval",
         instruction="Lower the left hand by 8 cm during the middle third."),
    dict(id="E05", source="squat", method="experimental", title="Crouch more deeply",
         instruction="Crouch lower, then return to the same standing pose.",
         text="A person performs a deep squat, lowering the pelvis close to the floor, then stands up.",
         body="lower_body"),
    dict(id="E06", source="jump", method="experimental", title="Jump higher",
         instruction="Perform the jumping jack with a higher jump.",
         text="A person performs a high jumping jack, jumping well above the ground, then returns to standing.",
         body="whole_body"),
    dict(id="E07", source="kick", method="experimental", title="Kick higher",
         instruction="Kick higher with the right leg, then return to standing.",
         text="A person performs a high forward kick with the right leg, then returns to standing.",
         body="right_leg"),
    dict(id="E08", source="bow", method="experimental", title="Bow more deeply",
         instruction="Bow more deeply, then return to the same standing pose.",
         text="A person bends the torso forward into a deep bow, then returns to standing.",
         body="torso"),
    dict(id="E09", source="kick", method="experimental", title="Switch the kicking side",
         instruction="Use the left leg for the forward kick instead of the right leg.",
         text="A person kicks forward with the left leg, then returns to standing.",
         body="lower_body"),
    dict(id="E10", source="squat", method="experimental", title="Make the squat shallower",
         instruction="Bend the knees less and keep the squat shallow.",
         text="A person performs a very shallow squat with a small bend of the knees, then stands upright.",
         body="lower_body"),
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def execute(command, output, gpu, log):
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    if not (output/"result.json").is_file():
        raise RuntimeError("Job returned without a result record")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-ref", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, default=[6, 7])
    parser.add_argument("--phase", choices=["generate", "edit"], required=True)
    args = parser.parse_args()
    import numpy as np
    from trace_core.references import read_poses, sha256
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    reference = read_poses(args.idle_ref)[0]
    if (output/"idle_pose35.npy").exists():
        if not np.array_equal(np.load(output/"idle_pose35.npy"), reference[None]):
            raise ValueError("Existing showcase uses a different idle reference")
    else:
        np.save(output/"idle_pose35.npy", reference[None])
        save(output/"manifest.json", dict(idle_source_sha256=sha256(args.idle_ref),
            idle_reference="first frame, identical at both ends", boundary_frames=8,
            sources=[dict(id=k, text=v) for k, v in SOURCES], cases=CASES,
            note="Actual inference examples; no success filtering and no benchmark statistics."))
    outcomes = []
    def worker(index):
        gpu = args.gpus[index]
        jobs = SOURCES[index::2] if args.phase == "generate" else (
            [CASES[i] for i in ([0, 3, 4, 6, 8] if index == 0 else [1, 2, 5, 7, 9])])
        local = []
        for job in jobs:
            name = job[0] if args.phase == "generate" else job["id"]
            directory = output/("sources" if args.phase == "generate" else "edits")/name
            if (directory/"result.json").is_file():
                if args.phase == "generate" or (directory/"comparison.mp4").is_file():
                    local.append(dict(id=name, status="existing"))
                    continue
            try:
                print(f"{args.phase} {name} on GPU {gpu}", flush=True)
                if args.phase == "generate":
                    command = [sys.executable, str(ROOT/"run.py"), "--text", job[1],
                        "--start-ref", str(output/"idle_pose35.npy"), "--end-ref", str(output/"idle_pose35.npy"),
                        "--start-frame", "0", "--end-frame", "0", "--contact-refine", "none",
                        "--render-device", str(gpu), "--output", str(directory)]
                else:
                    source = (output/"edits"/job["source"]/"motion_edited.sps"
                              if job["source"].startswith("E") else
                              output/"sources"/job["source"]/"motion_final.sps")
                    if job["method"] == "trained":
                        n = len(read_poses(source))
                        if not 32 <= n <= 104:
                            raise ValueError(f"Uncropped source length {n} outside this demo's editor range")
                        command = [sys.executable, str(ROOT/"edit.py"), "--source", str(source),
                            "--instruction", job["instruction"], "--render-device", str(gpu),
                            "--output", str(directory)]
                    else:
                        command = [sys.executable, str(ROOT/"tools/experimental_reference_edit.py"),
                            "--source", str(source), "--text", job["text"], "--instruction", job["instruction"],
                            "--body", job["body"], "--render-device", str(gpu), "--output", str(directory)]
                execute(command, directory, gpu, output/"logs"/f"{args.phase}_{name}.log")
                if args.phase == "edit":
                    edited = np.load(directory/"edited_pose35.npy")
                    source_pose = np.load(directory/"source_pose35.npy")
                    if not (np.array_equal(edited[:8], source_pose[:8]) and
                            np.array_equal(edited[-8:], source_pose[-8:])):
                        raise AssertionError("The edit changed protected idle frames")
                    env = os.environ.copy()
                    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
                    subprocess.run([sys.executable, str(ROOT/"tools/compare_edit.py"),
                                    "--input", str(directory)], cwd=ROOT, env=env, check=True)
                local.append(dict(id=name, status="complete", output=str(directory)))
                print(f"complete {name}", flush=True)
            except Exception as exc:
                local.append(dict(id=name, status="failed", error=str(exc)))
                print(f"failed {name}: {exc}", flush=True)
        return local
    with ThreadPoolExecutor(max_workers=2) as pool:
        for group in pool.map(worker, range(2)):
            outcomes.extend(group)
    save(output/f"{args.phase}_status.json", dict(time=time.time(), jobs=outcomes))
    if any(row["status"] == "failed" for row in outcomes):
        raise SystemExit("Some jobs failed; all statuses and logs were retained.")


if __name__ == "__main__":
    main()
