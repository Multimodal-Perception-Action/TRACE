"""Run self-authored natural-language generation trials and compose a review MP4."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = [
    dict(id="W01", title="Greeting a friend",
         text="A person notices a friend across the room and gives a big wave with the right arm."),
    dict(id="W02", title="Welcoming a guest",
         text="A person politely bows, then opens both arms to welcome a guest."),
    dict(id="W03", title="Making room",
         text="A person takes a small step sideways to the left to make room for someone."),
    dict(id="W04", title="Celebrating good news",
         text="A person celebrates good news by lifting both arms and making a small jump."),
    dict(id="W05", title="A startled reaction",
         text="A person leans backward and raises both forearms to protect the face, as if startled."),
    dict(id="W06", title="Inspecting the floor",
         text="A person leans forward to inspect something on the floor, then straightens up."),
    dict(id="W07", title="Balancing and lifting a knee",
         text="A person balances on the right leg and slowly lifts the left knee, then stands on both feet."),
    dict(id="W08", title="Stepping back and saying goodbye",
         text="A person takes two small steps backward, then waves goodbye with the left hand."),
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(output, idle_ref):
    from trace_core.references import sha256
    expected = dict(
        cases=CASES,
        reference_sha256=sha256(idle_ref),
        boundary="Same provided first-frame idle posture at both ends",
        steps=50, cfg=2.5, seed=42, contact_refine="heel_toe",
        protocol="Self-authored prompts; no training-set overlap audit; all trials retained.",
        visualization="Kinematic playback with smoothing and heel/toe refinement, not a physics rollout.",
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    manifest = output / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError("Existing manifest differs; use a new output directory")
    else:
        save(manifest, expected)


def generate(output, idle_ref, gpus):
    def worker(index):
        gpu = gpus[index]
        outcomes = []
        for case in CASES[index::len(gpus)]:
            directory = output / case["id"]
            try:
                if (directory / "result.json").is_file():
                    request = json.loads((directory / "request.json").read_text())
                    if request["text"] != case["text"]:
                        raise ValueError("Existing request has a different prompt")
                    if not (directory / "motion_smoothed.mp4").is_file():
                        raise ValueError("Existing inference has no rendered video")
                    outcomes.append(dict(id=case["id"], status="existing"))
                    continue
                if directory.exists():
                    raise FileExistsError(f"Incomplete output retained: {directory}")
                env = os.environ.copy()
                env.pop("PYTHONPATH", None)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                           OPENBLAS_NUM_THREADS="1", PYTHONUNBUFFERED="1")
                command = [
                    sys.executable, str(ROOT / "run.py"), "--text", case["text"],
                    "--start-ref", str(idle_ref), "--end-ref", str(idle_ref),
                    "--contact-refine", "heel_toe", "--render-device", str(gpu),
                    "--output", str(directory),
                ]
                print(f"Starting {case['id']} on GPU {gpu}: {case['title']}", flush=True)
                with (output / "logs" / f"{case['id']}.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                result = json.loads((directory / "result.json").read_text())
                if not result["success"] or not result["rendered"]:
                    raise ValueError("Inference/render did not complete")
                outcomes.append(dict(id=case["id"], status="complete",
                                     frames=result["frames"], seconds=result["seconds"]))
                print(f"Completed {case['id']}: {result['frames']} frames", flush=True)
            except Exception as exc:
                outcomes.append(dict(id=case["id"], status="failed", error=str(exc)))
                print(f"Failed {case['id']}: {exc}", flush=True)
        return outcomes

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        outcomes = [row for rows in pool.map(worker, range(len(gpus))) for row in rows]
    save(output / "status.json", dict(jobs=outcomes, time=time.time()))
    if any(row["status"] == "failed" for row in outcomes):
        raise RuntimeError("Some trials failed; retained all logs, without filtering cases")


def compose(output):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw
    from tools.compose_edit_showcase import font

    destination = output / "wild_examples.mp4"
    if destination.exists():
        raise FileExistsError(destination)
    for case in CASES:
        for filename in ("result.json", "motion_smoothed.mp4", "pose35_final.npy"):
            if not (output / case["id"] / filename).is_file():
                raise FileNotFoundError(output / case["id"] / filename)
    panels, timeline, total_frames = [], [], 0
    with imageio.get_writer(
        destination, fps=20, codec="libx264", quality=8, pixelformat="yuv420p",
        macro_block_size=2, output_params=["-movflags", "+faststart", "-threads", "2"],
    ) as writer:
        for case in CASES:
            directory = output / case["id"]
            result = json.loads((directory / "result.json").read_text())
            poses = np.load(directory / "pose35_final.npy", allow_pickle=False)
            if poses.shape != (result["frames"], 35) or not np.isfinite(poses).all():
                raise ValueError(f"Invalid motion array: {case['id']}")
            background = Image.new("RGB", (1280, 880), "#f5f8fc")
            draw = ImageDraw.Draw(background)
            draw.text((28, 18), f"TRACE  |  {case['id']}  {case['title']}",
                      font=font(27), fill="#172c42")
            for j, line in enumerate(textwrap.wrap(case["text"], 94)):
                draw.text((28, 57 + j * 27), line, font=font(23), fill="#283d50")
            draw.text((28, 844), "Kinematic preview | Fixed idle boundaries | Smoothing + heel/toe refinement",
                      font=font(18), fill="#52677b")
            timeline.append(dict(id=case["id"], text=case["text"],
                                 start_seconds=total_frames/20, duration_seconds=result["frames"]/20))
            reader = imageio.get_reader(directory / "motion_smoothed.mp4")
            count = 0
            try:
                for index, frame in enumerate(reader):
                    canvas = background.copy()
                    picture = Image.fromarray(frame)
                    picture.thumbnail((1280, 720), Image.Resampling.LANCZOS)
                    canvas.paste(picture, ((1280-picture.width)//2, 120+(720-picture.height)//2))
                    writer.append_data(np.asarray(canvas))
                    if index == result["frames"] // 2:
                        panels.append(canvas.copy())
                    count += 1
            finally:
                reader.close()
            if count != result["frames"]:
                raise ValueError(f"Video/motion frame mismatch: {case['id']}")
            total_frames += count
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(destination))
    if count != total_frames or abs(seconds-count/20) > .06:
        raise ValueError("Combined video duration mismatch")
    sheet = Image.new("RGB", (1280, 440*((len(panels)+1)//2)), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel.resize((640, 440), Image.Resampling.LANCZOS),
                    ((index % 2)*640, (index//2)*440))
    sheet.save(output / "overview.jpg", quality=92)
    save(output / "video_report.json", dict(
        file=destination.name, frames=count, fps=20, duration_seconds=seconds,
        width=1280, height=880, cases=timeline, no_success_filtering=True,
    ))
    print(f"Review video: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--idle-ref", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 4, 7])
    parser.add_argument("--phase", choices=["generate", "compose", "all"], default="all")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    output, idle_ref = args.output.resolve(), args.idle_ref.resolve()
    prepare(output, idle_ref)
    if args.phase in ("generate", "all"):
        generate(output, idle_ref, args.gpus)
    if args.phase in ("compose", "all"):
        compose(output)


if __name__ == "__main__":
    main()
