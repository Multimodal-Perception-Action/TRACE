"""Export bundled or user-provided G1 idle references as SPS and pose35 NPY."""
import argparse
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import read_poses
from trace_core.motion_io import write_sps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-ref", type=Path, default=ROOT / "examples/poses/idle_start.sps")
    parser.add_argument("--end-ref", type=Path, default=ROOT / "examples/poses/idle_end.sps")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/idle_examples")
    args = parser.parse_args()
    start = read_poses(args.start_ref)[0].copy()
    end = read_poses(args.end_ref)[0].copy()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (("idle_start", start), ("idle_end", end)):
        np.save(args.output / f"{name}.npy", value[None], allow_pickle=False)
        write_sps(args.output / f"{name}.sps", value[None], fps=20)
    print(args.output)


if __name__ == "__main__":
    main()
