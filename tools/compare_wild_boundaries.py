"""Matched inference without endpoint poses, with synchronized review videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import sha256, validate_checkpoint
from tools.run_wild_showcase import save

CONDITION_FILES = ("plan.json", "plan.npz", "text_embedding.npy", "planner_response.txt", "planner_status.json")


def generate(previous, output, cases, gpus):
    import numpy as np
    from trace_core.motion_io import write_sps

    def worker(index):
        gpu = gpus[index]
        outcomes = []
        for case in cases[index::len(gpus)]:
            source, destination = previous / case["id"], output / case["id"]
            started = time.perf_counter()
            try:
                if (destination / "result.json").exists():
                    outcomes.append(dict(id=case["id"], status="existing"))
                    continue
                config = json.loads((source / "request.json").read_text())
                original_checkpoint = json.loads((source / "checkpoint.json").read_text())
                current_checkpoint = validate_checkpoint(config["generator_checkpoint"], "free")
                if (config["text"] != case["text"] or config["boundary"] != "standing"
                        or current_checkpoint["sha256"] != original_checkpoint["sha256"]):
                    raise ValueError("Reference run or checkpoint changed")
                destination.mkdir()
                for key in ("boundary_reference_path", "boundary_reference_sha256"):
                    config.pop(key, None)
                config.update(boundary="free", boundary_frames=0, render_device=gpu,
                              matched_reference_run=str(source), conditioning_reused=True)
                save(destination / "request.json", config)
                save(destination / "checkpoint.json", current_checkpoint)
                hashes = {}
                for name in CONDITION_FILES:
                    shutil.copy2(source / name, destination / name)
                    hashes[name] = sha256(destination / name)
                    if hashes[name] != sha256(source / name):
                        raise ValueError(f"Copied condition changed: {name}")
                save(destination / "conditioning_provenance.json", dict(
                    source_run=str(source), sha256=hashes, regenerated=False))
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                print(f"Starting {case['id']} without boundaries on GPU {gpu}", flush=True)
                with (output / "logs" / f"{case['id']}.log").open("x") as log:
                    for stage in ("diffusion", "smooth", "contact"):
                        subprocess.run([
                            sys.executable, str(ROOT / "run.py"), "--stage", stage,
                            "--output", str(destination),
                        ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    # A union reference gives both panels the exact same fixed camera.
                    before = np.load(source / "pose35_final.npy", allow_pickle=False)
                    after = np.load(destination / "pose35_final.npy", allow_pickle=False)
                    if before.shape != after.shape or not np.isfinite(after).all():
                        raise ValueError("Motion length or finite-value mismatch")
                    if np.load(destination / "boundary_frame_mask.npy").any():
                        raise ValueError("Unexpected active endpoint mask")
                    camera = destination / "camera_reference.sps"
                    write_sps(camera, np.concatenate((before, after)), fps=20)
                    for name, motion in (("with_boundaries", source / "motion_final.sps"),
                                         ("without_boundaries", destination / "motion_final.sps")):
                        subprocess.run([
                            sys.executable, str(ROOT / "render.py"), "--input", str(motion),
                            "--output", str(destination / name), "--camera-reference", str(camera),
                            "--render-device", str(gpu), "--prompt", case["text"],
                        ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                save(destination / "result.json", dict(
                    success=True, frames=len(after), fps=20, duration_seconds=len(after)/20,
                    seconds=time.perf_counter()-started, boundary="free", boundary_frames=0,
                    generator_sha256=current_checkpoint["sha256"], conditioning_reused=True,
                    final_sps="motion_final.sps", video="without_boundaries/motion_smoothed.mp4",
                    contact_refine=config["contact_refine"], postprocessing=True,
                    shared_camera=True, physics_simulation=False,
                ))
                outcomes.append(dict(id=case["id"], status="complete", frames=len(after)))
                print(f"Completed {case['id']}", flush=True)
            except Exception as exc:
                outcomes.append(dict(id=case["id"], status="failed", error=str(exc)))
                print(f"Failed {case['id']}: {exc}", flush=True)
        return outcomes

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        outcomes = [row for rows in pool.map(worker, range(len(gpus))) for row in rows]
    save(output / "status.json", dict(jobs=outcomes))
    if any(row["status"] == "failed" for row in outcomes):
        raise RuntimeError("Some cases failed; all logs and partial results are retained")


def compose(output, cases):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw
    from tools.compose_edit_showcase import font

    comparison = output / "with_vs_without_boundaries.mp4"
    free_video = output / "without_boundaries.mp4"
    if comparison.exists() or free_video.exists():
        raise FileExistsError("Review videos already exist")
    for case in cases:
        if not (output / case["id"] / "result.json").is_file():
            raise FileNotFoundError(case["id"])
    options = dict(fps=20, codec="libx264", quality=8, pixelformat="yuv420p",
                   macro_block_size=2, output_params=["-movflags", "+faststart", "-threads", "2"])
    total, timeline, panels = 0, [], []
    with imageio.get_writer(comparison, **options) as paired, imageio.get_writer(free_video, **options) as single:
        for case in cases:
            directory = output / case["id"]
            n = json.loads((directory / "result.json").read_text())["frames"]
            card = Image.new("RGB", (1920, 920), "#f5f8fc")
            draw = ImageDraw.Draw(card)
            draw.text((30, 17), f"TRACE | {case['id']} | {case['title']}", font=font(29), fill="#182c43")
            for j, line in enumerate(textwrap.wrap(case["text"], 124)):
                draw.text((30, 59+j*29), line, font=font(25), fill="#293e53")
            draw.text((480, 144), "With start/end idle poses", font=font(28), fill="#17637b", anchor="mm")
            draw.text((1440, 144), "Without endpoint conditions", font=font(28), fill="#935b25", anchor="mm")
            draw.text((30, 893), "Same checkpoint, Plan, duration, text features and sampling settings | Kinematic playback with postprocessing",
                      font=font(21), fill="#52677b")
            solo = Image.new("RGB", (1280, 880), "#f5f8fc")
            sd = ImageDraw.Draw(solo)
            sd.text((28, 18), f"TRACE | {case['id']} | Without endpoint conditions", font=font(27), fill="#182c43")
            for j, line in enumerate(textwrap.wrap(case["text"], 94)):
                sd.text((28, 57+j*27), line, font=font(23), fill="#293e53")
            sd.text((28, 844), "Kinematic preview | Smoothing + heel/toe refinement", font=font(19), fill="#52677b")
            readers = [imageio.get_reader(directory / name / "motion_smoothed.mp4")
                       for name in ("with_boundaries", "without_boundaries")]
            count = 0
            try:
                from itertools import zip_longest
                for i, (before, after) in enumerate(zip_longest(*readers)):
                    if before is None or after is None:
                        raise ValueError("Rendered videos have different lengths")
                    canvas = card.copy()
                    canvas.paste(Image.fromarray(before), (0, 170))
                    canvas.paste(Image.fromarray(after), (960, 170))
                    paired.append_data(np.asarray(canvas))
                    only = solo.copy()
                    only.paste(Image.fromarray(after), (160, 120))
                    single.append_data(np.asarray(only))
                    if i == n // 2:
                        panels.append(canvas.resize((960, 460), Image.Resampling.LANCZOS))
                    count += 1
            finally:
                for reader in readers:
                    reader.close()
            if count != n:
                raise ValueError("Rendered and generated lengths differ")
            timeline.append(dict(id=case["id"], text=case["text"], start_seconds=total/20, duration_seconds=n/20))
            total += n
    reports = []
    for path in (comparison, free_video):
        count, duration = imageio_ffmpeg.count_frames_and_secs(str(path))
        if count != total or abs(duration-total/20) > .06:
            raise ValueError("Combined video length mismatch")
        reports.append(dict(file=path.name, frames=count, duration_seconds=duration))
    sheet = Image.new("RGB", (1920, 460*((len(panels)+1)//2)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i % 2)*960, (i//2)*460))
    sheet.save(output / "comparison_overview.jpg", quality=92)
    save(output / "video_report.json", dict(videos=reports, fps=20, cases=timeline, all_trials_retained=True))
    print(f"Comparison: {comparison}\nWithout boundaries: {free_video}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 4, 7])
    parser.add_argument("--phase", choices=["generate", "compose", "all"], default="all")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    previous, output = args.previous.resolve(), args.output.resolve()
    if previous == output:
        parser.error("Use a separate output folder")
    cases = json.loads((previous / "manifest.json").read_text())["cases"]
    expected = dict(cases=cases, source=str(previous),
                    comparison="Same boundary-capable weights; all endpoint masks disabled",
                    reuse="Exact Plan, frame count and text embedding from each previous run",
                    postprocessing="Same smoothing and contact solver; no protected endpoint frames in free mode")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()) != expected:
            raise ValueError("Existing manifest differs")
    else:
        save(output / "manifest.json", expected)
    if args.phase in ("generate", "all"):
        generate(previous, output, cases, args.gpus)
    if args.phase in ("compose", "all"):
        compose(output, cases)


if __name__ == "__main__":
    main()
