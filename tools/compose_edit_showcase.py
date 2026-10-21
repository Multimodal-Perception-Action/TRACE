"""Combine all completed review cases at their actual playback rate."""
import argparse
import json
from pathlib import Path
import textwrap

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve()
    manifest = json.loads((root/"manifest.json").read_text())
    targets = [root/"all_edit_examples.mp4", root/"trained_edit_examples.mp4",
               root/"exploratory_edit_examples.mp4"]
    if any(path.exists() for path in targets):
        raise FileExistsError("Review videos already exist")
    for case in manifest["cases"]:
        if not (root/"edits"/case["id"]/"comparison.mp4").is_file():
            raise FileNotFoundError(f"Incomplete case: {case['id']}; do not silently omit failed jobs")
    writers = [imageio.get_writer(path, fps=20, codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=2,
        output_params=["-movflags", "+faststart", "-threads", "2"]) for path in targets]
    panels = []
    try:
        for case in manifest["cases"]:
            label = f"{case['id']}  |  {case['title']}"
            method = ("Trained relative-hand editor" if case["method"] == "trained" else
                      "Exploratory reference resampling (not the trained editor)")
            card = Image.new("RGB", (1280, 720), "#f9fafc")
            draw = ImageDraw.Draw(card)
            draw.text((640, 240), label, fill="#1e334b", font=font(35), anchor="mm")
            for j, line in enumerate(textwrap.wrap(case["instruction"], 78)):
                draw.text((640, 310+j*35), line, fill="#26374a", font=font(26), anchor="mm")
            draw.text((640, 425), method, fill="#5e6d7b", font=font(22), anchor="mm")
            draw.text((640, 475), "Same idle pose at both ends | No postprocessing after editing",
                      fill="#5e6d7b", font=font(20), anchor="mm")
            selected = [writers[0], writers[1 if case["method"] == "trained" else 2]]
            for _ in range(20):
                for writer in selected:
                    writer.append_data(np.asarray(card))
            reader = imageio.get_reader(root/"edits"/case["id"]/"comparison.mp4")
            count = 0
            try:
                for index, frame in enumerate(reader):
                    canvas = Image.new("RGB", (1280, 720), "#f9fafc")
                    image = Image.fromarray(frame)
                    if image.width != 1280 or image.height > 664:
                        image.thumbnail((1280, 664), Image.Resampling.LANCZOS)
                    canvas.paste(image, ((1280-image.width)//2, 56+(664-image.height)//2))
                    ImageDraw.Draw(canvas).text((30, 28), label, fill="#1e334b", font=font(23), anchor="lm")
                    for writer in selected:
                        writer.append_data(np.asarray(canvas))
                    if index == 20:
                        panels.append(canvas.copy())
                    count += 1
            finally:
                reader.close()
            result = json.loads((root/"edits"/case["id"]/"result.json").read_text())
            if count != result["frames"]:
                raise ValueError(f"Wrong rendered duration for {case['id']}")
    finally:
        for writer in writers:
            writer.close()
    # A compact contact sheet helps locate cases; it is not a success ranking.
    sheet = Image.new("RGB", (1280, 360*((len(panels)+1)//2)), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel.resize((640, 360), Image.Resampling.LANCZOS),
                    ((i % 2)*640, (i//2)*360))
    sheet.save(root/"overview.jpg", quality=92)
    print("\n".join(str(path) for path in targets))


if __name__ == "__main__":
    main()
