"""Run the original ten user-study test prompts with the current release."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_core.references import sha256
from tools.run_readme_test_prompts import generate, compose
from tools.run_wild_showcase import save


def load_cases(survey, comparison):
    """Cross-check prompt identity and order against both archived manifests."""
    original = json.loads(survey.read_text(encoding="utf-8"))
    standard = [row for row in original if row["kind"] == "standard"]
    archive = json.loads(comparison.read_text(encoding="utf-8"))
    if archive["cohort"] != "original_user_study_standard_10":
        raise ValueError("Unexpected comparison cohort")
    if sha256(survey) != archive["survey_source_sha256"]:
        raise ValueError("Original survey differs from its archived fingerprint")
    expected = sorted(archive["samples"], key=lambda row: row["survey_order"])
    if len(standard) != 10 or len(expected) != 10:
        raise ValueError("Expected exactly ten original test cases")
    if len({row["uid"] for row in standard}) != 10:
        raise ValueError("Duplicate survey case")
    cases = []
    for index, (source, matched) in enumerate(zip(standard, expected), 1):
        if (source["uid"], source["text"]) != (matched["sample_id"], matched["text"]):
            raise ValueError(f"Prompt or order mismatch at position {index}")
        cases.append(dict(
            id=source["uid"], title=f"User study test {index:02d}/10",
            text=source["text"], text_zh=source.get("zh", ""),
            motion_uid=matched["uid"], survey_order=index,
        ))
    return cases


def prepare(output, survey, comparison, boundary="standing"):
    if boundary not in ("standing", "free"):
        raise ValueError("Expected standing/free boundary mode")
    cases = load_cases(survey, comparison)
    output.mkdir(parents=True, exist_ok=True)
    (output/"logs").mkdir(exist_ok=True)
    reference = ROOT/"models/standing_pose35.npy"
    manifest = dict(
        cases=cases, cohort="original_user_study_standard_10",
        survey_source=str(survey), survey_source_sha256=sha256(survey),
        comparison_source=str(comparison), comparison_source_sha256=sha256(comparison),
        new_inference=True, ground_truth_motion_used=False,
        duration_source="Fresh LLM prediction; no supplied or GT duration",
        checkpoint=str(ROOT/"models/generator_boundary.pt"),
        checkpoint_sha256=sha256(ROOT/"models/generator_boundary.pt"),
        idle_reference_sha256=sha256(reference) if boundary == "standing" else None,
        boundary=("Same current reference first frame at start and end" if boundary == "standing"
                  else "No start/end pose inputs or boundary projections; same generator weights"),
        steps=50, cfg=2.5, seed=42, precision="bf16",
        postprocessing="Smoothing and heel/toe four-point refinement",
        physical_robot_rollout=False, all_requested_cases_retained=True,
        note="Fresh visualization of the same prompts, not new human-study scores.",
    )
    local_reference = output/"idle_pose35.npy"
    if (output/"manifest.json").exists():
        if json.loads((output/"manifest.json").read_text()) != manifest:
            raise ValueError("Existing manifest differs; use a new directory")
        if boundary == "standing" and sha256(local_reference) != manifest["idle_reference_sha256"]:
            raise ValueError("Idle reference changed")
    else:
        if boundary == "standing":
            shutil.copy2(reference, local_reference)
        save(output/"manifest.json", manifest)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survey", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--phase", choices=["generate", "compose", "all"], default="all")
    parser.add_argument("--boundary", choices=["standing", "free"], default="standing")
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or any(g < 0 for g in args.gpus):
        parser.error("GPU indices must be unique and nonnegative")
    output = args.output.resolve()
    cases = prepare(output, args.survey.resolve(), args.comparison.resolve(), args.boundary)
    if args.phase in ("generate", "all"):
        generate(output, args.gpus, cases, args.boundary)
    if args.phase in ("compose", "all"):
        filename = "user_study_test10_free.mp4" if args.boundary == "free" else "user_study_test10.mp4"
        compose(output, cases, filename=filename)


if __name__ == "__main__":
    main()
