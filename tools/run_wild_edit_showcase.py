"""Edit generated wild-prompt motions and retain all trials for visual review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import sha256
from tools.run_edit_showcase import execute, save

CASES = [
    dict(id="WE01", source="W01", method="trained", title="Greeting: lower the right hand",
         instruction="Lower the right hand by 10 cm during the middle half."),
    dict(id="WE02", source="W01", method="trained", title="Greeting: raise the right hand",
         instruction="Raise the right hand by 8 cm during the middle half."),
    dict(id="WE03", source="W05", method="trained", title="Protect the face: raise the left hand",
         instruction="Raise the left hand by 8 cm during the middle half."),
    dict(id="WE04", source="W08", method="trained", title="Goodbye: lower the left hand",
         instruction="Lower the left hand by 10 cm during the middle half."),
    dict(id="WE05", source="W06", method="experimental", title="Inspect the floor: bend more deeply",
         instruction="Bend the torso farther forward while inspecting the floor.",
         text="A person bends the torso deeply forward to look closely at the floor, then straightens up.",
         body="torso"),
    dict(id="WE06", source="W06", method="experimental", title="Inspect the floor: a smaller bend",
         instruction="Bend the torso less and stay more upright while inspecting the floor.",
         text="A person makes a small shallow forward lean to glance at the floor, then stands upright.",
         body="torso"),
    dict(id="WE07", source="W07", method="experimental", title="Balance: lift the left knee higher",
         instruction="Lift the left knee higher toward the chest.",
         text="A person balances on the right leg and slowly lifts the left knee high toward the chest, then stands on both feet.",
         body="left_leg"),
    dict(id="WE08", source="W04", method="experimental", title="Celebrate with a higher jump",
         instruction="Jump higher while celebrating with both arms raised.",
         text="A person celebrates good news by raising both arms and jumping high above the ground.",
         body="whole_body"),
]


def prepare(previous, output):
    source_cases = {row["id"]: row for row in json.loads((previous / "manifest.json").read_text())["cases"]}
    sources = []
    for key in sorted({case["source"] for case in CASES}):
        folder = previous / key
        result = json.loads((folder / "result.json").read_text())
        if result["boundary"] != "free" or not result["success"]:
            raise ValueError("Expected completed no-endpoint-condition source motions")
        sources.append(dict(id=key, text=source_cases[key]["text"], frames=result["frames"],
                            sps_sha256=sha256(folder / "motion_final.sps")))
    manifest = dict(
        sources=sources, cases=CASES, source_directory=str(previous),
        source_postprocessing="Smoothing and heel/toe contact refinement",
        editing_postprocessing=False, editing_ik=False,
        boundary="No new idle poses; each edit retains the actual source endpoints",
        methods=dict(trained="Separate trained hand-height editor with trained LLM scope parser",
                     experimental="Generator reference resampling with manually specified body scope"),
        protocol="All planned trials retained; pipeline completion does not imply edit success",
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in ("edits", "logs", "overlays"):
        (output / name).mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a separate output directory")
    else:
        save(output / "manifest.json", manifest)


def generate(previous, output, gpus):
    import numpy as np
    from trace_core.references import read_poses
    expected_sources = {row["id"]: row for row in json.loads((output / "manifest.json").read_text())["sources"]}

    def worker(index):
        gpu = gpus[index]
        outcomes = []
        for case in CASES[index::len(gpus)]:
            directory = output / "edits" / case["id"]
            source = previous / case["source"] / "motion_final.sps"
            try:
                if sha256(source) != expected_sources[case["source"]]["sps_sha256"]:
                    raise ValueError("Source motion changed")
                if not (directory / "result.json").is_file():
                    if directory.exists():
                        raise FileExistsError(f"Incomplete edit retained: {directory}")
                    if case["method"] == "trained":
                        if not 32 <= len(read_poses(source)) <= 104:
                            raise ValueError("Uncropped source exceeds the trained editor's supported length")
                        command = [sys.executable, str(ROOT / "edit.py"),
                                   "--source", str(source), "--instruction", case["instruction"],
                                   "--render-device", str(gpu), "--output", str(directory)]
                    else:
                        command = [sys.executable, str(ROOT / "tools/experimental_reference_edit.py"),
                                   "--source", str(source), "--instruction", case["instruction"],
                                   "--text", case["text"], "--body", case["body"],
                                   "--render-device", str(gpu), "--output", str(directory)]
                    print(f"Editing {case['id']} from {case['source']} on GPU {gpu}", flush=True)
                    execute(command, directory, gpu, output / "logs" / f"{case['id']}.log")
                before = np.load(directory / "source_pose35.npy", allow_pickle=False)
                after = np.load(directory / "edited_pose35.npy", allow_pickle=False)
                np.testing.assert_array_equal(before, read_poses(source))
                if before.shape != after.shape or not np.isfinite(after).all():
                    raise ValueError("Invalid edited motion shape or values")
                np.testing.assert_array_equal(before[:8], after[:8])
                np.testing.assert_array_equal(before[-8:], after[-8:])
                env = os.environ.copy()
                env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2", CUDA_VISIBLE_DEVICES=str(gpu))
                with (output / "logs" / f"{case['id']}_visualization.log").open("a") as log:
                    if not (directory / "comparison.mp4").is_file():
                        subprocess.run([sys.executable, str(ROOT / "tools/compare_edit.py"),
                                        "--input", str(directory)], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    subprocess.run([
                        sys.executable, str(ROOT / "tools/render_edit_overlay.py"),
                        "--input", str(output), "--output", str(output / "overlays"),
                        "--case", case["id"], "--device", str(gpu),
                        "--priority", "before", "--no-compose", "--resume",
                    ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                outcomes.append(dict(id=case["id"], status="complete", frames=len(after),
                                     method=case["method"], original_endpoints_preserved=True))
                print(f"Completed {case['id']}", flush=True)
            except Exception as exc:
                outcomes.append(dict(id=case["id"], status="failed", error=str(exc)))
                print(f"Failed {case['id']}: {exc}", flush=True)
        return outcomes

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        outcomes = [row for rows in pool.map(worker, range(len(gpus))) for row in rows]
    save(output / "status.json", dict(jobs=outcomes))
    if any(row["status"] == "failed" for row in outcomes):
        raise RuntimeError("Some trials failed; all outputs and logs have been retained")


def compose(output):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw
    from tools.compose_edit_showcase import font
    from tools.render_edit_overlay import compose as compose_overlays

    source_texts = {row["id"]: row["text"] for row in json.loads((output / "manifest.json").read_text())["sources"]}
    for case in CASES:
        if not (output / "edits" / case["id"] / "comparison.mp4").is_file():
            raise FileNotFoundError(case["id"])
    targets = [output / "all_edit_examples.mp4", output / "trained_hand_edits.mp4",
               output / "exploratory_body_edits.mp4"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Combined review videos already exist")
    writers = [imageio.get_writer(path, fps=20, codec="libx264", quality=8,
               pixelformat="yuv420p", macro_block_size=2,
               output_params=["-movflags", "+faststart", "-threads", "2"]) for path in targets]
    counts, timeline = [0, 0, 0], []
    try:
        for case in CASES:
            directory = output / "edits" / case["id"]
            result = json.loads((directory / "result.json").read_text())
            background = Image.new("RGB", (1280, 800), "#f9fafc")
            draw = ImageDraw.Draw(background)
            draw.text((25, 18), f"{case['id']} | {case['title']}", font=font(26), fill="#1e334b")
            for j, line in enumerate(textwrap.wrap("Source: " + source_texts[case["source"]], 101)):
                draw.text((25, 56+27*j), line, font=font(21), fill="#465c73")
            selected = (0, 1 if case["method"] == "trained" else 2)
            timeline.append(dict(id=case["id"], source=case["source"], method=case["method"],
                                 instruction=case["instruction"], start_seconds=counts[0]/20,
                                 duration_seconds=result["frames"]/20))
            reader = imageio.get_reader(directory / "comparison.mp4")
            count = 0
            try:
                for frame in reader:
                    canvas = background.copy()
                    panel = Image.fromarray(frame)
                    panel.thumbnail((1280, 680), Image.Resampling.LANCZOS)
                    canvas.paste(panel, ((1280-panel.width)//2, 120+(680-panel.height)//2))
                    for index in selected:
                        writers[index].append_data(np.asarray(canvas))
                        counts[index] += 1
                    count += 1
            finally:
                reader.close()
            if count != result["frames"]:
                raise ValueError("Before/after video frame count differs from the edited motion")
    finally:
        for writer in writers:
            writer.close()
    reports = []
    for path, expected in zip(targets, counts):
        count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
        if count != expected or abs(duration-count/20) > .06:
            raise ValueError("Combined video frame count or duration differs")
        reports.append(dict(file=path.name, frames=count, fps=20, duration_seconds=duration))
    compose_overlays(output / "overlays", CASES)
    save(output / "video_report.json", dict(videos=reports, cases=timeline, all_trials_retained=True))
    print("\n".join(str(path) for path in targets), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 4, 7])
    parser.add_argument("--phase", choices=["generate", "compose", "all"], default="all")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    previous, output = args.previous.resolve(), args.output.resolve()
    if previous == output:
        parser.error("Use a new output directory")
    prepare(previous, output)
    if args.phase in ("generate", "all"):
        generate(previous, output, args.gpus)
    if args.phase in ("compose", "all"):
        compose(output)


if __name__ == "__main__":
    main()
