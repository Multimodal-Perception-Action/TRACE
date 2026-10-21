"""Render source, previous edit and a measured jump-guided candidate."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import read_poses, sha256
from trace_core.motion_io import write_sps
from tools.optimize_jump_demo import measure
from tools.run_edit_showcase import save


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 4, 7])
    args = parser.parse_args()
    previous, candidate, output = [p.resolve() for p in (args.previous, args.candidate, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    source = previous/"source.sps"
    paths = [source, previous/"motion_edited.sps", candidate/"motion_edited.sps"]
    names = ["source", "previous", "improved"]
    poses = [read_poses(p) for p in paths]
    if len(set(p.shape for p in poses)) != 1:
        raise ValueError("Cannot synchronize different motion shapes")
    camera = output/"camera_reference.sps"
    write_sps(camera, np.concatenate(poses), fps=20)
    rows = [dict(id=name, path=str(p), sha256=sha256(p), metrics=measure(p, source))
            for name, p in zip(names, paths)]
    case = dict(id="JUMP", title="A higher celebration jump",
                instruction="Jump higher while keeping the arm gesture.",
                method="experimental", method_label="Jump-target-guided sampling")
    save(output/"manifest.json", dict(
        cases=[case], motions=rows, inference_result=json.loads((candidate/"result.json").read_text()),
        note="Actual model resampling plus numerical FK guidance; no dynamics validation."))
    folder = output/"edits"/"JUMP"
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, folder/"source.sps")
    shutil.copy2(candidate/"motion_edited.sps", folder/"motion_edited.sps")
    jobs = []
    for i, (name, path) in enumerate(zip(names, paths)):
        jobs.append((name, [sys.executable, str(ROOT/"render.py"), "--input", str(path),
                          "--output", str(output/name), "--camera-reference", str(camera),
                          "--render-device", str(args.gpus[i % len(args.gpus)]),
                          "--prompt", case["instruction"]]))
    jobs.append(("overlay", [sys.executable, str(ROOT/"tools/render_edit_overlay.py"),
                            "--input", str(output), "--output", str(output/"overlays"),
                            "--device", str(args.gpus[-1]), "--priority", "before",
                            "--no-compose", "--resume"]))
    def run(job):
        name, command = job
        expected = output/name/"motion_smoothed.mp4"
        if name != "overlay" and expected.is_file():
            return
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
        with (output/"logs"/f"{name}.log").open("a") as log:
            subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print("Rendered "+name, flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        list(pool.map(run, jobs))
    compose(output, rows)


def compose(output, rows):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw
    from tools.render_edit_overlay import font
    video = output/"jump_comparison.mp4"
    if video.exists():
        raise FileExistsError(video)
    readers = [imageio.get_reader(output/row["id"]/"motion_smoothed.mp4") for row in rows]
    headers = ["Original motion", "Previous edit", "Improved edit"]
    colors = ["#246bcd", "#798492", "#e66b35"]
    canvas = Image.new("RGB", (1920, 690), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((960, 24), '"Jump higher while keeping the arm gesture."',
              font=font(28, True), fill="#202f44", anchor="mt")
    draw.text((960, 68), "Same source · Same frame · Same camera · Real-time playback",
              font=font(21), fill="#58677d", anchor="mt")
    for i, (title, color, row) in enumerate(zip(headers, colors, rows)):
        x = 320+640*i
        draw.text((x, 111), title, font=font(25, True), fill=color, anchor="mt")
        draw.text((x, 148), f"Peak sole clearance: {row['metrics']['both_feet_peak_cm']:.1f} cm",
                  font=font(22), fill="#334155", anchor="mt")
    draw.text((960, 671), "Improved: numerical jump targets inside denoising; no post-smoothing. Kinematic playback.",
              font=font(19), fill="#58677d", anchor="mm")
    target_frames = rows[0]["metrics"]["frames"]
    key_frames = [8, 11, 14, 16, 18, 20, 22, 24, 28, 35]
    stills = []
    try:
        with imageio.get_writer(video, fps=20, codec="libx264", quality=8,
                pixelformat="yuv420p", macro_block_size=2,
                output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
            for index in range(target_frames):
                picture = canvas.copy()
                for i, reader in enumerate(readers):
                    image = Image.fromarray(reader.get_next_data()).resize((640, 480), Image.Resampling.LANCZOS)
                    picture.paste(image, (640*i, 180))
                writer.append_data(np.asarray(picture))
                if index == rows[2]["metrics"]["peak_frame"]:
                    picture.save(output/"comparison.png")
                if index in key_frames:
                    stills.append(picture.resize((960, 345), Image.Resampling.LANCZOS))
    finally:
        for reader in readers:
            reader.close()
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if count != target_frames:
        raise ValueError("Comparison frame count mismatch")
    sheet = Image.new("RGB", (1920, 345*((len(stills)+1)//2)), "white")
    for i, image in enumerate(stills):
        sheet.paste(image, ((i%2)*960, (i//2)*345))
    sheet.save(output/"contact_sheet.jpg", quality=92)
    save(output/"video_check.json", dict(frames=count, seconds=seconds, fps=20,
                                        retimed=False, completed=True))
    print(video, flush=True)


if __name__ == "__main__":
    main()
