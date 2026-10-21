"""Compose synchronized before/after editor renders without temporal resampling."""
import argparse
from itertools import zip_longest
import json
from pathlib import Path
import textwrap

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Completed edit output directory")
    args = parser.parse_args()
    root = args.input.resolve()
    target = root / "comparison.mp4"
    if target.exists() or (root / "comparison.png").exists():
        raise FileExistsError("Comparison output exists; existing results are preserved")
    request = json.loads((root / "request.json").read_text())
    result = json.loads((root / "result.json").read_text())
    readers = [imageio.get_reader(root / name / "motion_smoothed.mp4")
               for name in ("before", "after")]
    try:
        for reader in readers:
            if abs(reader.get_meta_data()["fps"] - result["fps"]) > 1e-6:
                raise ValueError("Rendered FPS differs from motion FPS")
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 24)
            small = ImageFont.truetype("DejaVuSans.ttf", 20)
        except OSError:
            font = ImageFont.load_default(size=24)
            small = ImageFont.load_default(size=20)
        lines = textwrap.wrap(request["instruction"], width=91)
        header = 104 + 30 * len(lines)
        height = header + 480
        height += height % 2
        canvas = Image.new("RGB", (1280, height), "#f9fafc")
        draw = ImageDraw.Draw(canvas)
        for i, line in enumerate(lines):
            draw.text((640, 22 + 30*i), line, fill="#202a38", font=font, anchor="mt")
        method = ("Exploratory reference resampling" if
                  result.get("method") == "experimental_generator_reference_resampling"
                  else "Trained relative-hand editor")
        draw.text((640, header-66), method, fill="#586679", font=small, anchor="mt")
        for x, title in ((320, "Before"), (960, "After")):
            draw.text((x, header-30), title, fill="#202a38", font=small, anchor="mm")
        count = 0
        with imageio.get_writer(target, fps=result["fps"], codec="libx264",
                quality=8, pixelformat="yuv420p", macro_block_size=2,
                output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
            for i, pair in enumerate(zip_longest(*readers)):
                if any(frame is None for frame in pair):
                    raise ValueError("Before/after videos have different frame counts")
                frame = canvas.copy()
                for j, image in enumerate(pair):
                    frame.paste(Image.fromarray(image).resize((640, 480), Image.Resampling.LANCZOS),
                                (j*640, header))
                writer.append_data(np.asarray(frame))
                if i == result["frames"] // 2:
                    frame.save(root / "comparison.png")
                count += 1
        if count != result["frames"]:
            raise ValueError("Rendered frame count differs from source motion")
        print(target)
    finally:
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()
