"""Replay the archived paper jumping-jack edit without new inference or repair."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import read_poses, sha256
from tools.render_edit_overlay import save
from tools.optimize_jump_demo import measure


def main():
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    archive, output = args.archive.resolve(), args.output.resolve()
    context = json.loads((archive/"context.json").read_text())
    records = [json.loads(line) for line in
               (archive/"edited_motion/predictions.jsonl").read_text().splitlines() if line.strip()]
    selected = [row for row in records if row["motion_uid"] == context["motion_uid"]]
    if len(selected) != 1:
        raise ValueError("Expected exactly one recorded paper edit")
    row = selected[0]
    source, edited = Path(context["base_sps"]), Path(row["generated_sps"])
    if Path(row["base_sps"]).resolve() != source.resolve():
        raise ValueError("Source provenance differs between metadata files")
    before, after = read_poses(source), read_poses(edited)
    if before.shape != after.shape or len(before) != row["num_frames"]:
        raise ValueError("Archived source and edit have different lengths")
    if row["fps"] != 20:
        raise ValueError("Expected the archived 20 FPS motion")
    output.mkdir(parents=True, exist_ok=False)
    folder = output/"edits"/"PAPER_JUMP"
    folder.mkdir(parents=True)
    for path, filename in ((source, "source.sps"), (edited, "motion_edited.sps")):
        shutil.copy2(path, folder/filename)
        if sha256(path) != sha256(folder/filename):
            raise ValueError("Copied SPS differs from the archived source")
    case = dict(
        id="PAPER_JUMP", title="Jumping jack: jump higher",
        instruction='"Perform a jumping jack."  ->  "Can you jump higher?"',
        method="archived_paper_edit", method_label="Original paper demo",
    )
    save(output/"manifest.json", dict(
        cases=[case], archive=str(archive), source_path=str(source), edited_path=str(edited),
        source_sha256=sha256(source), edited_sha256=sha256(edited),
        source_prompt=context["base_text_zh"], feedback=context["feedback"],
        checkpoint=row["checkpoint"], repair_mode=row["repair_mode"],
        frames=row["num_frames"], fps=row["fps"],
        rerender_only=True, new_inference=False, height_adjustment=False,
        new_smoothing=False, temporal_resampling=False, physical_rollout=False,
        before_metrics=measure(source), after_metrics=measure(edited, source),
    ))
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2")
    with (output/"render.log").open("x") as log:
        subprocess.run([
            sys.executable, str(ROOT/"tools/render_edit_overlay.py"),
            "--input", str(output), "--output", str(output/"overlays"),
            "--device", str(args.device), "--priority", "before", "--no-compose",
        ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from tools.render_edit_overlay import font
    video = output/"overlays/PAPER_JUMP/overlay.mp4"
    indices = sorted(set(np.linspace(0, len(before)-1, 6).round().astype(int).tolist()
                         +[int(after[:, 1].argmax()), int(after[:, 1].argmin())]))
    sheet = Image.new("RGB", (1280, 430*((len(indices)+1)//2)), "#f8fafc")
    reader = imageio.get_reader(video)
    try:
        count, position = 0, 0
        for i, frame in enumerate(reader):
            count += 1
            if i not in indices:
                continue
            tile = Image.new("RGB", (640, 430), "#f8fafc")
            tile.paste(Image.fromarray(frame).resize((640, 400), Image.Resampling.LANCZOS))
            ImageDraw.Draw(tile).text((14, 406), f"Frame {i+1} / {len(before)}",
                                      font=font(17), fill="#334155")
            sheet.paste(tile, ((position%2)*640, (position//2)*430))
            position += 1
    finally:
        reader.close()
    if count != len(before):
        raise ValueError("Rendered video has missing or extra frames")
    sheet.save(output/"sequence_preview.jpg", quality=94)
    save(output/"verification.json", dict(
        frames=count, fps=20, duration_seconds=count/20,
        original_motion_files_unchanged=(
            sha256(source) == sha256(folder/"source.sps")
            and sha256(edited) == sha256(folder/"motion_edited.sps")),
        complete=True,
    ))
    print(video, flush=True)


if __name__ == "__main__":
    main()
