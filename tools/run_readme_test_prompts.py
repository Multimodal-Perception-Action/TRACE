"""Generate fresh motions from the three Test prompts in the README robot demo."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import read_poses, sha256
from tools.run_wild_showcase import save

CASES = [
    dict(id="T01", title="Bodyweight groove",
         text="grooving by shifting bodyweight from one leg to another, hands on hips to hip hop music"),
    dict(id="T02", title="Lock and kick",
         text="a strong dancer performs a series of lock and kick moves to the rhythm of hip-hop"),
    dict(id="T03", title="Hip-hop: kick, strike, return",
         text="person quickly performs a hip-hop dance routine that includes a leg kick, arm strike, and return"),
]


def prepare(output):
    output.mkdir(parents=True, exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    reference = ROOT/"models/standing_pose35.npy"
    local_reference = output/"idle_pose35.npy"
    manifest = dict(
        cases=CASES,
        prompt_source="README.md > Real-robot demonstrations > Test prompts segment (06_test.mp4)",
        source_demo="media/real_robot_demo.mp4",
        new_inference=True, ground_truth_motion_used=False,
        duration_source="Fresh LLM prediction; no supplied or GT duration",
        checkpoint=str(ROOT/"models/generator_boundary.pt"),
        checkpoint_sha256=sha256(ROOT/"models/generator_boundary.pt"),
        idle_reference_sha256=sha256(reference),
        boundary="Same current reference first frame at start and end",
        steps=50, cfg=2.5, seed=42, precision="bf16",
        postprocessing="Smoothing and heel/toe four-point refinement",
        physical_robot_rollout=False, all_requested_cases_retained=True,
    )
    if (output/"manifest.json").exists():
        if json.loads((output/"manifest.json").read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new directory")
        if sha256(local_reference) != manifest["idle_reference_sha256"]:
            raise ValueError("Idle reference changed")
    else:
        shutil.copy2(reference, local_reference)
        save(output/"manifest.json", manifest)


def generation_command(output, case, gpu, boundary="standing"):
    """Use the same checkpoint; free mode never passes pose references."""
    if boundary not in ("standing", "free"):
        raise ValueError("This comparison supports standing/free on the same model")
    command = [
        sys.executable, str(ROOT/"run.py"), "--text", case["text"],
        "--boundary", boundary, "--contact-refine", "heel_toe",
        "--precision", "bf16", "--steps", "50", "--cfg", "2.5",
        "--seed", "42", "--render-device", str(gpu),
        "--output", str(output/case["id"]),
    ]
    if boundary == "standing":
        command += [
            "--start-ref", str(output/"idle_pose35.npy"),
            "--end-ref", str(output/"idle_pose35.npy"),
        ]
    return command


def generate(output, gpus, cases=None, boundary="standing"):
    cases = CASES if cases is None else cases
    def worker(index):
        gpu = gpus[index]
        outcomes = []
        for case in cases[index::len(gpus)]:
            folder = output/case["id"]
            try:
                if folder.exists():
                    raise FileExistsError(f"Preserving existing output: {folder}")
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                command = generation_command(output, case, gpu, boundary)
                print(f"Starting {case['id']} on GPU {gpu}: {case['text']}", flush=True)
                with (output/"logs"/f"{case['id']}.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                result = json.loads((folder/"result.json").read_text())
                if not result["success"] or not result["rendered"]:
                    raise RuntimeError("Incomplete generation/render")
                if json.loads((folder/"request.json").read_text())["text"] != case["text"]:
                    raise ValueError("Prompt differs from demo")
                if result["boundary"] != boundary:
                    raise ValueError("Unexpected boundary mode")
                pose = read_poses(folder/"motion_final.sps")
                if len(pose) != result["frames"]:
                    raise ValueError("Incorrect SPS frame count")
                outcomes.append(dict(id=case["id"], status="complete",
                                     frames=result["frames"], seconds=result["seconds"]))
                print(f"Completed {case['id']}: {result['frames']} frames", flush=True)
            except Exception as exc:
                outcomes.append(dict(id=case["id"], status="failed", error=str(exc)))
                print(f"Failed {case['id']}: {exc}", flush=True)
        return outcomes
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        outcomes = [row for rows in pool.map(worker, range(len(gpus))) for row in rows]
    save(output/"status.json", dict(jobs=outcomes))
    if any(row["status"] == "failed" for row in outcomes):
        raise RuntimeError("Some requested cases failed; all logs and outputs retained")


def compose(output, cases=None, filename="readme_test_prompts.mp4"):
    cases = CASES if cases is None else cases
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw
    from tools.render_edit_overlay import font
    target = output/filename
    if target.exists():
        raise FileExistsError(target)
    frames, timeline, panels = 0, [], []
    with imageio.get_writer(target, fps=20, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=2,
            output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
        for case in cases:
            folder = output/case["id"]
            result = json.loads((folder/"result.json").read_text())
            canvas = Image.new("RGB", (1280, 880), "#f5f8fc")
            draw = ImageDraw.Draw(canvas)
            draw.text((28, 18), f"TRACE  |  {case['id']}  {case['title']}",
                      font=font(27, True), fill="#172c42")
            for i, line in enumerate(textwrap.wrap(case["text"], 93)):
                draw.text((28, 58+27*i), line, font=font(22), fill="#334155")
            boundary_label = ("Fixed idle boundaries" if result["boundary"] == "standing"
                              else "No start/end pose constraints")
            draw.text((28, 844), f"Fresh inference | {boundary_label} | Smoothing + heel/toe refinement | Kinematic replay",
                      font=font(18), fill="#52677b")
            reader = imageio.get_reader(folder/"motion_smoothed.mp4")
            count = 0
            preview = None
            try:
                if reader.get_meta_data()["fps"] != 20:
                    raise ValueError("Unexpected render frame rate")
                for index, rgb in enumerate(reader):
                    frame = canvas.copy()
                    picture = Image.fromarray(rgb)
                    picture.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                    frame.paste(picture, ((1280-picture.width)//2, 120+(720-picture.height)//2))
                    writer.append_data(np.asarray(frame))
                    if index == result["frames"]//2:
                        preview = frame.copy()
                    count += 1
            finally:
                reader.close()
            if count != result["frames"] or preview is None:
                raise ValueError("Incomplete video")
            panels.append(preview)
            timeline.append(dict(id=case["id"], text=case["text"],
                                 start_seconds=frames/20, duration_seconds=count/20))
            frames += count
    actual, seconds = imageio_ffmpeg.count_frames_and_secs(str(target))
    if actual != frames:
        raise ValueError("Combined video frame count mismatch")
    columns = 3 if len(panels) <= 3 else 2
    rows = (len(panels) + columns - 1)//columns
    sheet = Image.new("RGB", (640*columns, 440*rows), "#f5f8fc")
    for i, panel in enumerate(panels):
        sheet.paste(panel.resize((640, 440), Image.Resampling.LANCZOS),
                    (640*(i % columns), 440*(i // columns)))
    sheet.save(output/"overview.jpg", quality=94)
    save(output/"video_report.json", dict(frames=actual, fps=20, duration_seconds=seconds,
                                         timeline=timeline, all_requested_cases_retained=True))
    print(target, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 4])
    parser.add_argument("--phase", choices=["generate", "compose", "all"], default="all")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    output = args.output.resolve()
    prepare(output)
    if args.phase in ("generate", "all"):
        generate(output, args.gpus)
    if args.phase in ("compose", "all"):
        compose(output)


if __name__ == "__main__":
    main()
