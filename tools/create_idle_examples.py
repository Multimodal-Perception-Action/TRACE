"""Create two synthetic idle references from the distributed default G1 pose."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import read_poses
from trace_core.motion_io import SPS_JOINTS, write_sps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/idle_examples")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = read_poses(args.models / "standing_pose35.npy")[0].copy()
    end = start.copy()
    # Two relaxed poses, not an extracted internal mocap recording.
    for name in ("left_elbow_joint", "right_elbow_joint"):
        end[6 + SPS_JOINTS[1:].index(name)] = 45.
    for name, value in (("idle_start", start), ("idle_end", end)):
        np.save(args.output / f"{name}.npy", value[None], allow_pickle=False)
        write_sps(args.output / f"{name}.sps", value[None], fps=20)
    print(args.output)


if __name__ == "__main__":
    main()
