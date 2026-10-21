"""Audit and compare real jump-edit inference candidates; keep every trial."""
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
from tools.run_edit_showcase import save


def measure(path, source=None):
    import numpy as np
    from trace_core.foot_labels import FootGeometry, runs
    pose = read_poses(path)
    geometry = FootGeometry()
    points, _ = geometry.sequence(pose)
    clearance = points[..., 2].min((1, 2))
    air = clearance > .02
    report = dict(
        frames=len(pose), fps=20, both_feet_peak_cm=float(clearance.max()*100),
        peak_frame=int(clearance.argmax()), airborne_spans=runs(air),
        airborne_seconds=float(air.sum()/20), root_peak_cm=float(pose[:, 1].max()),
        penetration_max_cm=float(max(0, -clearance.min())*100),
        penetration_mean_cm=float(np.maximum(-points[..., 2], 0).mean()*100),
        root_acceleration_rms_cm_s2=float(np.sqrt(np.mean((np.diff(pose[:, 1], n=2)*400)**2))),
        knee_peak_deg=np.max(pose[:, [9, 15]], axis=0).tolist(),
        leg_joint_limit_excess_deg=float(np.rad2deg(
            np.maximum(geometry.limits[:12, 0]-np.deg2rad(pose[:, 6:18]), 0)
            +np.maximum(np.deg2rad(pose[:, 6:18])-geometry.limits[:12, 1], 0)).max()),
    )
    if source is not None:
        base = read_poses(source)
        np.testing.assert_equal(pose.shape, base.shape)
        report.update(
            upper_body_exact=bool(np.array_equal(pose[:, 18:], base[:, 18:])),
            arm_joints_exact=bool(np.array_equal(pose[:, 21:], base[:, 21:])),
            endpoint_frames_exact=bool(np.array_equal(pose[:8], base[:8]) and np.array_equal(pose[-8:], base[-8:])),
            root_xy_exact=bool(np.array_equal(pose[:, [0, 2]], base[:, [0, 2]])),
            root_orientation_exact=bool(np.array_equal(pose[:, 3:6], base[:, 3:6])),
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 2, 4, 7])
    parser.add_argument("--phase", choices=["sweep", "guided"], default="sweep")
    args = parser.parse_args()
    previous, output = args.previous.resolve(), args.output.resolve()
    source = previous / "source.sps"
    config = json.loads((previous / "request.json").read_text())
    source_metrics = measure(source)
    peak = source_metrics["peak_frame"]
    start, end = max(8, peak-10), min(source_metrics["frames"]-8, peak+18)
    if args.phase == "sweep":
        candidates = [
            dict(id="S65_C3", strength=.65, cfg=3.),
            dict(id="S80_C3", strength=.8, cfg=3.),
            dict(id="S65_C5", strength=.65, cfg=5.),
            dict(id="S80_C5", strength=.8, cfg=5.),
        ]
    else:
        candidates = [
            dict(id="G10_v2", strength=.65, cfg=3., height_gain_cm=10.),
            dict(id="G14_v2", strength=.65, cfg=3., height_gain_cm=14.),
            dict(id="G18_v2", strength=.65, cfg=3., height_gain_cm=18.),
        ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    save(output / "baseline_metrics.json", dict(source=source_metrics, previous_edit=measure(previous/"motion_edited.sps", source)))
    save(output / f"{args.phase}_manifest.json", dict(
        source=str(source), source_sha256=sha256(source), previous_edit=str(previous),
        active_span=[start, end], candidates=candidates,
        selection="Geometric demo tuning; not a held-out benchmark or physical execution claim",
    ))

    def worker(index):
        rows = []
        gpu = args.gpus[index]
        for candidate in candidates[index::len(args.gpus)]:
            folder = output / candidate["id"]
            try:
                if not (folder / "result.json").exists():
                    folder.mkdir(exist_ok=True)
                    for name in ("source.sps", "source_pose35.npy", "text_embedding.npy",
                                 "plan.npz", "plan.json", "planner_response.txt"):
                        shutil.copy2(previous/name, folder/name)
                    settings = dict(config, **{k: v for k, v in candidate.items() if k != "id"})
                    settings.update(body="lower_body", active_span=[start, end])
                    save(folder / "request.json", settings)
                    env = os.environ.copy()
                    env.pop("PYTHONPATH", None)
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
                    print(f"Running {candidate['id']} on GPU {gpu}", flush=True)
                    with (output / "logs" / f"{candidate['id']}.log").open("x") as log:
                        subprocess.run([sys.executable, str(ROOT/"tools/experimental_reference_edit.py"),
                                        "--stage", "infer", "--output", str(folder)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                row = dict(id=candidate["id"], status="complete", metrics=measure(folder/"motion_edited.sps", source))
                save(folder/"jump_metrics.json", row["metrics"])
                print(json.dumps(row), flush=True)
                rows.append(row)
            except Exception as exc:
                rows.append(dict(id=candidate["id"], status="failed", error=str(exc)))
                print(f"Failed {candidate['id']}: {exc}", flush=True)
        return rows
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        rows = [row for group in pool.map(worker, range(len(args.gpus))) for row in group]
    save(output / f"{args.phase}_results.json", dict(candidates=rows))
    if any(row["status"] == "failed" for row in rows):
        raise SystemExit("Some candidates failed; logs retained")


if __name__ == "__main__":
    main()
