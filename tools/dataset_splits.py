"""Export filename-only splits and resolve them against a local motion archive.

This tool does not download or upload data, copy motion files, or retain source
captions/metadata. It preserves split membership and row order.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

SPLITS = ("train", "val", "test")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def valid_name(value):
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value
            or any(ord(c) < 32 for c in value) or not Path(value).suffix):
        raise ValueError("Expected a filename with an extension, without a directory")
    return value


def read_splits(directory, source=False):
    result, owners, families = {}, {}, {}
    for split in SPLITS:
        names = []
        with (Path(directory) / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    raise ValueError(f"Blank row in {split}:{line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object in {split}:{line_number}")
                if source:
                    path = row.get("filepath")
                    if not isinstance(path, str) or "\\" in path:
                        raise ValueError(f"Invalid source filepath in {split}:{line_number}")
                    name = valid_name(Path(path).name)
                    metadata_name = row.get("metadata", {}).get("filename")
                    if metadata_name is not None and metadata_name not in (name, Path(name).stem):
                        raise ValueError(f"Conflicting filename in {split}:{line_number}")
                else:
                    if set(row) != {"filename"}:
                        raise ValueError(f"Only the filename field is allowed in {split}:{line_number}")
                    name = valid_name(row["filename"])
                if name in owners:
                    raise ValueError(f"Duplicate filename: {name} ({owners[name]}, {split})")
                owners[name] = split
                # BONES-SEED sagittal mirrors use a terminal _M suffix.
                stem = Path(name).stem
                family = stem[:-2] if stem.endswith("_M") else stem
                if family in families and families[family] != split:
                    raise ValueError(f"Mirror pair crosses splits: {family}")
                families[family] = split
                names.append(name)
        if not names:
            raise ValueError(f"Empty split: {split}")
        result[split] = names
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write("\n")


def export_splits(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    before = {s: digest(source_dir / f"{s}.jsonl") for s in SPLITS}
    splits = read_splits(source_dir, source=True)
    after = {s: digest(source_dir / f"{s}.jsonl") for s in SPLITS}
    if before != after:
        raise ValueError("Input manifests changed during export")
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, names in splits.items():
        with (output_dir / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for name in names:
                stream.write(json.dumps({"filename": name}, ensure_ascii=True) + "\n")
    report = {
        "schema_version": 1,
        "fields": ["filename"],
        "counts": {s: len(names) for s, names in splits.items()},
        "total": sum(map(len, splits.values())),
        "source_sha256": before,
        "sha256": {s: digest(output_dir / f"{s}.jsonl") for s in SPLITS},
        "source_order_preserved": True,
        "unique_filenames": True,
        "cross_split_mirror_groups": 0,
        "includes_motions": False,
        "includes_captions_or_actor_metadata": False,
    }
    write_json(output_dir / "manifest.json", report)
    return report


def resolve_splits(split_dir, motion_root, output_dir, motion_suffix=None):
    splits = read_splits(split_dir)
    motion_root, output_dir = Path(motion_root).resolve(), Path(output_dir)
    if not motion_root.is_dir():
        raise ValueError("Motion root is not a directory")
    suffixes = (".sps", ".csv", ".npy", ".npz")
    if motion_suffix is not None and motion_suffix not in suffixes:
        raise ValueError("Unsupported motion suffix")
    wanted = {n: Path(n).with_suffix(motion_suffix).name if motion_suffix else n
              for names in splits.values() for n in names}
    if len(set(wanted.values())) != len(wanted):
        raise ValueError("Suffix mapping creates ambiguous filenames")
    needed, found = set(wanted.values()), {}
    for parent, directories, filenames in os.walk(motion_root, followlinks=False):
        directories[:] = [n for n in directories if not (Path(parent) / n).is_symlink()]
        for name in filenames:
            if name not in needed:
                continue
            path = Path(parent) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Expected a regular motion file: {name}")
            if name in found:
                raise ValueError(f"Ambiguous filename in motion archive: {name}")
            found[name] = path.relative_to(motion_root).as_posix()
    missing = needed - found.keys()
    if missing:
        raise ValueError(f"Missing {len(missing)} motion files; first examples: {sorted(missing)[:5]}")
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, names in splits.items():
        with (output_dir / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for name in names:
                stream.write(json.dumps({"filename": name, "motion_path": found[wanted[name]]}) + "\n")
    report = {"counts": {s: len(v) for s, v in splits.items()},
              "paths_relative_to_motion_root": True, "motion_suffix": motion_suffix,
              "motions_copied_or_converted": False}
    write_json(output_dir / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Remove all fields except the motion filename")
    export.add_argument("--source-dir", type=Path, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    resolve = commands.add_parser("resolve", help="Locate each filename in a local motion tree")
    resolve.add_argument("--splits", type=Path, required=True)
    resolve.add_argument("--motion-root", type=Path, required=True)
    resolve.add_argument("--output-dir", type=Path, required=True)
    resolve.add_argument("--motion-suffix", choices=[".sps", ".csv", ".npy", ".npz"])
    args = parser.parse_args()
    try:
        if args.command == "export":
            result = export_splits(args.source_dir, args.output_dir)
        else:
            result = resolve_splits(args.splits, args.motion_root, args.output_dir, args.motion_suffix)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
