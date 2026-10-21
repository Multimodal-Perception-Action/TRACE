"""Render actual before/after motions together without moving or retiming them."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BEFORE = (0.12, 0.42, 0.82, 1.0)
AFTER = (0.94, 0.35, 0.12, 0.35)
VIEW = 640


def colors(priority):
    return (BEFORE, AFTER) if priority == "before" else (
        (*BEFORE[:3], 0.30), (*AFTER[:3], 1.0))


def overlay_xml(priority="before"):
    """Instantiate two visual skeletons sharing assets and world coordinates."""
    from render import scene_xml

    root = ET.fromstring(scene_xml(VIEW, VIEW))
    # Only kinematic playback is needed. Remove references to the original names.
    for node in list(root):
        if node.tag not in {
            "compiler", "option", "size", "visual", "statistic",
            "default", "asset", "worldbody",
        }:
            root.remove(node)
    world = root.find("worldbody")
    templates = list(world.findall("body"))
    if len(templates) != 1:
        raise ValueError("Expected one source robot")
    world.remove(templates[0])
    for prefix, color in zip(("before", "after"), colors(priority)):
        body = copy.deepcopy(templates[0])
        for node in body.iter():
            if "name" in node.attrib:
                node.set("name", prefix + "_" + node.get("name"))
            if node.tag == "geom":
                node.attrib.pop("material", None)
                node.set("rgba", " ".join(map(str, color)))
                node.set("contype", "0")
                node.set("conaffinity", "0")
        world.append(body)
    for texture in root.findall("asset/texture"):
        if texture.get("name") == "floor_tex":
            texture.set("rgb1", "0.88 0.91 0.94")
            texture.set("rgb2", "0.96 0.97 0.98")
    for material in root.findall("asset/material"):
        if material.get("name") == "floor_mat":
            material.set("reflectance", "0")
    return ET.tostring(root, encoding="unicode")


def font(size, bold=False):
    from PIL import ImageFont
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(name, size)


def header(case, priority):
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1280, 800), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((28, 16), f"{case['id']}  {case['title']}",
              font=font(25, True), fill="#1e293b")
    for row, line in enumerate(textwrap.wrap(case["instruction"], 88)):
        draw.text((28, 53 + 25 * row), line, font=font(21), fill="#475569")
    for x, color, label in (
        (28, "#246bcd", "Before · priority" if priority == "before" else "Before · transparent"),
        (335, "#f05a1f", "After · transparent" if priority == "before" else "After · priority"),
    ):
        draw.rounded_rectangle((x, 96, x + 16, 112), radius=4, fill=color)
        draw.text((x + 25, 94), label, font=font(18), fill="#334155")
    method = case.get("method_label") or (
        "Trained hand editor" if case["method"] == "trained"
        else "Exploratory reference resampling")
    draw.text((1252, 101), method, font=font(17), fill="#64748b", anchor="rm")
    draw.text((28, 779), "Same frame · Same world coordinates · Original playback speed",
              font=font(16), fill="#64748b", anchor="lm")
    return canvas


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def composite_priority(combined, foreground, segmentation, geom_ids, geom_type):
    """Keep the selected robot's pixels intact even when the other copy is nearer."""
    import numpy as np
    mask = (segmentation[..., 1] == geom_type) & np.isin(segmentation[..., 0], geom_ids)
    result = combined.copy()
    result[mask] = foreground[mask]
    return result


def render_case(root, case, output, device, still_only=False, priority="before", edit_directory=None):
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device)
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import mujoco as mj
    import numpy as np
    from PIL import Image, ImageDraw
    from render import load_motion
    from trace_core.motion_io import SPS_JOINTS

    start = time.monotonic()
    folder = Path(edit_directory) if edit_directory is not None else root / "edits" / case["id"]
    paths = [folder / "source.sps", folder / "motion_edited.sps"]
    sequences = [load_motion(path) for path in paths]
    (before, fps), (after, other_fps) = sequences
    if fps != other_fps or before.shape != after.shape:
        raise ValueError("Overlay requires exactly aligned motions; no retiming is allowed")
    output.mkdir(parents=True, exist_ok=True)
    target = output / case["id"]
    target.mkdir(exist_ok=True)
    product = target / ("preview.png" if still_only else "overlay.mp4")
    if product.exists():
        raise FileExistsError(product)
    model = mj.MjModel.from_xml_string(overlay_xml(priority))
    data = mj.MjData(model)
    addresses, roots, body_groups = [], [], []
    for prefix in ("before", "after"):
        joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{prefix}_{name}")
                     for name in SPS_JOINTS[1:]]
        free = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix + "_pelvis")
        if min(joint_ids + [free]) < 0:
            raise ValueError("Missing overlay robot joint")
        addresses.append(model.jnt_qposadr[joint_ids])
        roots.append(int(model.jnt_qposadr[free]))
        body_groups.append([i for i in range(model.nbody)
                            if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or "")
                            .startswith(prefix + "_")])
    if len(body_groups[0]) != len(body_groups[1]):
        raise ValueError("Mismatched skeletons")
    primary = 0 if priority == "before" else 1
    secondary = 1 - primary
    palette = colors(priority)
    secondary_geoms = [np.flatnonzero(model.geom_bodyid == body)
                       for body in body_groups[secondary]]
    primary_geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, body_groups[primary]))
    secondary_geom_ids = np.flatnonzero(np.isin(model.geom_bodyid, body_groups[secondary]))

    def set_frame(index):
        for sequence, adr, offset in zip((before, after), addresses, roots):
            data.qpos[offset:offset + 7] = sequence[index, :7]
            data.qpos[adr] = sequence[index, 7:]
        data.qvel[:] = 0
        mj.mj_forward(model, data)
        # Exact overlap needs one surface, not two coplanar surfaces (z-fighting).
        # This changes visibility only, never either motion or its coordinates.
        for before_body, after_body, geoms in zip(*body_groups, secondary_geoms):
            identical = (np.allclose(data.xpos[before_body], data.xpos[after_body],
                                     rtol=0, atol=1e-9) and
                         np.allclose(data.xmat[before_body], data.xmat[after_body],
                                     rtol=0, atol=1e-9))
            model.geom_rgba[geoms, 3] = 0 if identical else palette[secondary][3]

    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    differences = []
    for i in range(len(before)):
        set_frame(i)
        low = np.minimum(low, data.xpos[1:].min(0))
        high = np.maximum(high, data.xpos[1:].max(0))
        differences.append(float(np.linalg.norm(
            data.xpos[body_groups[0]] - data.xpos[body_groups[1]], axis=-1).max()))
    # The still uses the largest visible displacement; the video includes every frame.
    representative = int(np.argmax(differences))
    low -= .16
    high += .16
    cameras = []
    for angle in (135, 90):
        camera = mj.MjvCamera()
        mj.mjv_defaultCamera(camera)
        camera.lookat[:] = (low + high) / 2
        camera.distance = max(2.0, np.linalg.norm(high - low) / 2 /
                              math.sin(math.radians(float(model.vis.global_.fovy) / 2)) * 1.04)
        camera.azimuth, camera.elevation = angle, -12
        cameras.append(camera)
    options = mj.MjvOption()
    options.geomgroup[0] = 0
    options.sitegroup[:] = 0
    model.geom_group[mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "floor")] = 2
    renderer = mj.Renderer(model, height=VIEW, width=VIEW)
    background = header(case, priority)

    def make_frame(index):
        set_frame(index)
        canvas = background.copy()
        for view, camera in enumerate(cameras):
            renderer.update_scene(data, camera=camera, scene_option=options)
            # Shadows from two coincident copies obscure the floor/contact comparison.
            renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
            rgb = renderer.render()
            # Render the priority copy alone, then restore its pixels on top.
            # Screen-space overlap follows the requested visibility priority;
            # depth, FK, positions and frame alignment of both motions stay unchanged.
            alpha = model.geom_rgba[secondary_geom_ids, 3].copy()
            model.geom_rgba[secondary_geom_ids, 3] = 0
            try:
                renderer.update_scene(data, camera=camera, scene_option=options)
                renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                foreground = renderer.render()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render()
                renderer.disable_segmentation_rendering()
                rgb = composite_priority(rgb, foreground, segmentation, primary_geom_ids,
                                         int(mj.mjtObj.mjOBJ_GEOM))
            finally:
                renderer.disable_segmentation_rendering()
                model.geom_rgba[secondary_geom_ids, 3] = alpha
            canvas.paste(Image.fromarray(rgb), (view * VIEW, 120))
        draw = ImageDraw.Draw(canvas)
        for x, text in ((20, "Three-quarter view"), (660, "Side view")):
            draw.rounded_rectangle((x, 136, x + 205, 166), radius=7, fill="#ffffff")
            draw.text((x + 10, 141), text, font=font(16), fill="#475569")
        draw.line((640, 120, 640, 760), fill="#e2e8f0", width=2)
        draw.text((1252, 779), f"{index + 1} / {len(before)}",
                  font=font(16), fill="#64748b", anchor="rm")
        return canvas

    temporary = target / ".overlay.partial.mp4"
    try:
        make_frame(representative).save(target / "preview.png")
        if not still_only:
            with imageio.get_writer(str(temporary), fps=fps, codec="libx264",
                    pixelformat="yuv420p", macro_block_size=1, quality=8,
                    ffmpeg_log_level="error",
                    output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
                for index in range(len(before)):
                    writer.append_data(np.asarray(make_frame(index)))
            count, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
            if count != len(before) or abs(duration - count / fps) > 1 / fps + .01:
                raise RuntimeError("Video timing verification failed")
            temporary.replace(product)
    finally:
        renderer.close()
    report = dict(case=case["id"], frames=len(before), fps=fps, method=case["method"],
        source_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],
        colors=dict(before=palette[0], after=palette[1]), shared_world_coordinates=True,
        overlap_priority=priority, screen_space_priority=True,
        coincident_surfaces_hidden=("after" if priority == "before" else "before"),
        position_offsets=False, temporal_resampling=False, joint_clipping=False,
        postprocessing=False, physics_simulation=False, view_azimuths=[135, 90],
        still_frame=representative, still_selection="maximum body-position difference",
        seconds=time.monotonic() - start, complete=not still_only)
    save(target / "render_report.json", report)
    print(json.dumps(report), flush=True)


def compose(output, cases):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image

    target = output / "all_edit_overlays.mp4"
    if target.exists():
        raise FileExistsError(target)
    expected = 0
    previews = []
    temporary = output / ".all_edit_overlays.partial.mp4"
    with imageio.get_writer(str(temporary), fps=20, codec="libx264", quality=8,
            pixelformat="yuv420p", macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-movflags", "+faststart", "-threads", "2"]) as writer:
        for case in cases:
            folder = output / case["id"]
            report = json.loads((folder / "render_report.json").read_text())
            if not report["complete"] or report["fps"] != 20:
                raise ValueError(f"Incomplete case: {case['id']}")
            count = 0
            reader = imageio.get_reader(str(folder / "overlay.mp4"))
            try:
                for frame in reader:
                    writer.append_data(frame)
                    count += 1
            finally:
                reader.close()
            if count != report["frames"]:
                raise ValueError("Missing frames")
            expected += count
            with Image.open(folder / "preview.png") as preview:
                previews.append(preview.copy().resize((640, 400), Image.Resampling.LANCZOS))
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(temporary))
    if count != expected:
        raise ValueError("Combined video dropped frames")
    temporary.replace(target)
    sheet = Image.new("RGB", (1280, 400 * math.ceil(len(cases) / 2)), "white")
    for i, preview in enumerate(previews):
        sheet.paste(preview, ((i % 2) * 640, (i // 2) * 400))
    sheet.save(output / "overview.jpg", quality=93)
    save(output / "manifest.json", dict(cases=cases, frames=count, duration=duration,
        video=target.name, fps=20, no_retiming=True, no_new_inference=True))
    print(str(target), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Showcase directory with a case manifest")
    source.add_argument("--edit", type=Path, help="One completed edit directory; no case manifest needed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Render only one case ID")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--still-only", action="store_true")
    parser.add_argument("--compose-only", action="store_true")
    parser.add_argument("--no-compose", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse completed overlays only when inputs and priority match")
    parser.add_argument("--priority", choices=("before", "after"), default="before",
                        help="Show this robot in front wherever silhouettes overlap")
    args = parser.parse_args()
    if args.edit:
        directory = args.edit.resolve()
        request = json.loads((directory / "request.json").read_text())
        result = json.loads((directory / "result.json").read_text())
        experimental = result.get("method") == "experimental_generator_reference_resampling"
        case = dict(id=directory.name, title="Motion editing", instruction=request["instruction"],
                    method="experimental" if experimental else "trained")
        if result.get("numerical_guidance"):
            case["method_label"] = "Reference-guided generation + height target"
        cases = [case] if not args.case or args.case == case["id"] else []
        root = directory.parent
    else:
        root = args.input.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        cases = [case for case in manifest["cases"] if not args.case or case["id"] == args.case]
    if not cases:
        parser.error("No matching cases")
    if not args.compose_only:
        for case in cases:
            report_path = args.output / case["id"] / "render_report.json"
            if args.resume and not args.still_only and report_path.exists():
                report = json.loads(report_path.read_text())
                folder = args.edit if args.edit else root / "edits" / case["id"]
                hashes = [hashlib.sha256((folder / name).read_bytes()).hexdigest()
                          for name in ("source.sps", "motion_edited.sps")]
                if (report.get("complete") and report.get("overlap_priority") == args.priority
                        and report.get("source_sha256") == hashes
                        and (report_path.parent / "overlay.mp4").is_file()):
                    print(f"Verified existing overlay: {case['id']}", flush=True)
                    continue
                raise ValueError(f"Existing overlay does not match current request: {case['id']}")
            render_case(root, case, args.output.resolve(),
                        args.device, args.still_only, args.priority, edit_directory=args.edit)
    if not args.still_only and not args.case and not args.no_compose:
        compose(args.output.resolve(), cases)


if __name__ == "__main__":
    main()
