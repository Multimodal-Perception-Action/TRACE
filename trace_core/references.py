"""Validated G1 pose references and compatible generator checkpoint selection."""
from pathlib import Path
import hashlib
import re

import numpy as np
from .motion_io import SPS_JOINTS, convert_to_35dim, parse_sps_file, to_gravity_gauge


def resolve_idle_references(start=None, end=None):
    """Use the bundled idle SPS for each omitted endpoint independently."""
    directory = Path(__file__).resolve().parents[1] / "examples/poses"
    return (Path(start) if start is not None else directory / "idle_start.sps",
            Path(end) if end is not None else directory / "idle_end.sps")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_poses(path):
    """Read finite pose35 arrays or named-joint SPS; never invent missing joints."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        poses = np.load(path, allow_pickle=False)
        if poses.shape == (35,):
            poses = poses[None]
    elif path.suffix.lower() == ".sps":
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"^\s*(\d+)\s*$", text, flags=re.M)
        indices = [int(v) for v in blocks[1::2]]
        if not indices or indices != list(range(indices[0], indices[0] + len(indices))):
            raise ValueError("SPS frame indices must be consecutive")
        for block in blocks[2::2]:
            names = [line.split()[0] for line in block.splitlines()
                     if line.strip() and not line.lstrip().startswith("#")]
            if any(names.count(name) != 1 for name in SPS_JOINTS):
                raise ValueError("SPS must contain root and all 29 named G1 joints exactly once")
            if re.search(r"\b(?:nan|inf(?:inity)?)\b", block, flags=re.I):
                raise ValueError("Non-finite SPS values")
        poses = convert_to_35dim(parse_sps_file(path))
    else:
        raise ValueError("References accept .sps or .npy pose35 arrays")
    if poses.ndim != 2 or poses.shape[1] != 35 or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("Expected a nonempty finite [T,35] G1 pose array")
    return np.asarray(poses, np.float32)


def select_pose(path, frame=0):
    poses = read_poses(path)
    if type(frame) is not int or not -len(poses) <= frame < len(poses):
        raise ValueError("Reference frame index is out of range")
    return poses[frame].copy()


def prepare_references(default, start=None, end=None, start_frame=0, end_frame=0):
    if not (start and end):
        default = read_poses(default)
        if default.shape != (2, 35):
            raise ValueError("Default idle reference must contain start and end poses [2,35]")
    poses = np.stack((
        select_pose(start, start_frame) if start else default[0],
        select_pose(end, end_frame) if end else default[1],
    ))
    return poses


def boundary_condition(poses, frames, width, mean, std):
    """Encode endpoints separately: do not create a displacement between two idles.

    Height, yaw-free tilt and joints are conditioned. Planar increments and yaw
    increments are zero in each idle band; absolute world XY/yaw are not targets.
    """
    poses = np.asarray(poses, np.float32)
    if poses.shape != (2, 35) or not np.isfinite(poses).all():
        raise ValueError("Expected finite start/end references [2,35]")
    mean, std = np.asarray(mean), np.asarray(std)
    if (mean.shape != (40,) or std.shape != (40,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("Boundary normalization requires finite 40D mean and positive std")
    if type(width) is not int or not 1 <= width <= 16:
        raise ValueError("Idle band width must be an integer in [1,16]")
    if frames < 2 * width + 4:
        raise ValueError("Motion is too short for the requested idle bands")
    reference = np.zeros((1, frames, 40), np.float32)
    known = np.zeros_like(reference, dtype=bool)
    for pose, span in ((poses[0], slice(0, width)), (poses[1], slice(frames-width, frames))):
        state = to_gravity_gauge(np.repeat(pose[None], 2, axis=0))[0]
        reference[0, span] = (state - mean) / std
        known[0, span] = True
    return reference, known


def validate_checkpoint(path, boundary):
    """Only explicitly supported inference exports are accepted; no silent coercion."""
    import torch
    if boundary not in ("standing", "free", "none"):
        raise ValueError("Unknown boundary mode")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = "boundary_events_v1" if boundary in ("standing", "free") else "sparse_events_v1"
    if checkpoint.get("format") != expected:
        raise ValueError(f"Expected {expected}; select a compatible checkpoint or change --boundary")
    config = checkpoint["model_config"]
    if (config["feature_dim"], config["condition_embedding_dim"], config["patch_size"],
            config["max_patches"]) != (40, 2560, 4, 128):
        raise ValueError("Checkpoint does not match the released G1 event/embedding interface")
    if "ema" not in checkpoint:
        raise ValueError("Checkpoint must contain EMA inference weights")
    return {"file": Path(path).name, "sha256": sha256(path), "format": expected,
            "model_config": config, "step": checkpoint.get("step")}
