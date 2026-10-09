#!/usr/bin/env python3
"""Transfer the reviewed release assets through the official ``ms-hub`` CLI.

Install ``modelscope-hub`` separately. Authentication is delegated to its normal
login flow; this helper never reads credentials or accepts tokens. Uploads are
dry runs unless both --execute and the manifest's approval flag are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import subprocess
import sys
import tempfile


REPO_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")
COMPONENT_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
METADATA_FILES = {
    "README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE_PENDING.md",
    "NOTICE", "NOTICE.md", "THIRD_PARTY_NOTICES.md", "configuration.json",
    ".gitattributes",
}
OFFICIAL_UPSTREAM = {
    "Qwen/Qwen3-4B-Instruct-2507": "models/planner_base",
    "Qwen/Qwen3-Embedding-4B": "models/text_embedding",
}
UPSTREAM_FILES = (
    "LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICE.md",
    "*.safetensors", "model.safetensors.index.json", "config.json",
    "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
    "chat_template.jinja",
    "tokenizer.model", "sentencepiece.bpe.model", "modules.json",
    "sentence_bert_config.json", "config_sentence_transformers.json",
    "1_Pooling/config.json",
)


class AssetError(ValueError):
    """An unsafe or inconsistent release was refused."""


def validate_repo_id(value: str) -> str:
    if not isinstance(value, str) or not REPO_PATTERN.fullmatch(value):
        raise AssetError(f"Expected a repository ID in owner/name form: {value!r}")
    return value


def relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise AssetError(f"Invalid relative path: {value!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} or not COMPONENT_PATTERN.fullmatch(part)
           or part.startswith("-") for part in parts):
        raise AssetError(f"Unsafe relative path: {value!r}")
    return PurePosixPath(value)


def validate_revision(value: str) -> str:
    relative_path(value)
    if value.startswith("."):
        raise AssetError(f"Invalid revision: {value!r}")
    return value


def no_symlinks(path: Path) -> None:
    """Reject symlinks in the path and all its ancestors, including broken links."""
    for item in (path, *path.parents):
        if item.is_symlink():
            raise AssetError(f"Symlinks are not allowed: {item}")
    for parent in path.parents:
        if parent.exists() and not parent.is_dir():
            raise AssetError(f"Expected a parent directory: {parent}")


def safe_path(root: Path, relative: str) -> Path:
    path = root.joinpath(*relative_path(relative).parts)
    no_symlinks(path)
    return path


def regular_file(path: Path) -> None:
    no_symlinks(path)
    if not path.exists() or not stat.S_ISREG(path.stat().st_mode):
        raise AssetError(f"Expected a regular file: {path}")


def sha256(path: Path) -> str:
    regular_file(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: dict) -> None:
    regular_file(path)
    if path.stat().st_size != artifact["bytes"] or sha256(path) != artifact["sha256"]:
        raise AssetError(f"Size or SHA256 mismatch; refusing to use or overwrite {path}")


def read_manifest(root: Path) -> dict:
    path = safe_path(root, "release_manifest.json")
    regular_file(path)
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise AssetError("Manifest must be an object")
    if type(manifest.get("public_release_approved")) is not bool:
        raise AssetError("Manifest must contain a boolean public_release_approved")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise AssetError("Manifest artifacts must be a nonempty list")
    destinations = set()
    sources = set()
    for item in artifacts:
        if not isinstance(item, dict) or item.get("repo_type") not in ("model", "dataset"):
            raise AssetError("Every artifact needs repo_type=model or dataset")
        relative_path(item.get("path"))
        relative_path(item.get("destination"))
        destination = "models/" + item["path"]
        if item["repo_type"] == "dataset" and item["path"] == "example_prompts.jsonl":
            destination = "data/example_prompts.jsonl"
        if item["destination"] != destination:
            raise AssetError(f"Unexpected destination for {item['path']}; expected {destination}")
        if type(item.get("bytes")) is not int or item["bytes"] < 0:
            raise AssetError("Artifact bytes must be a nonnegative integer")
        if not isinstance(item.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"]):
            raise AssetError("Artifact sha256 must contain 64 lowercase hexadecimal digits")
        source = (item["repo_type"], item["path"])
        if source in sources or item["destination"] in destinations:
            raise AssetError("Duplicate artifact path or destination")
        sources.add(source)
        destinations.add(item["destination"])
    upstream = manifest.get("upstream", [])
    if not isinstance(upstream, list):
        raise AssetError("Manifest upstream must be a list")
    upstream_ids = set()
    for item in upstream:
        if (not isinstance(item, dict) or not isinstance(item.get("repo_id"), str)
                or item["repo_id"] not in OFFICIAL_UPSTREAM):
            raise AssetError("Only the reviewed official Qwen upstream repositories are allowed")
        if item.get("destination") != OFFICIAL_UPSTREAM[item["repo_id"]]:
            raise AssetError("Unexpected upstream destination")
        validate_revision(item.get("revision", "master"))
        if item["repo_id"] in upstream_ids:
            raise AssetError("Duplicate upstream repository")
        if item["destination"] in destinations:
            raise AssetError("Artifact and upstream destinations overlap")
        upstream_ids.add(item["repo_id"])
        destinations.add(item["destination"])
    for path in destinations:
        if any(str(parent) in destinations for parent in PurePosixPath(path).parents):
            raise AssetError("Manifest destinations overlap")
    return manifest


def scan_files(directory: Path) -> list[Path]:
    no_symlinks(directory)
    if not directory.is_dir():
        raise AssetError(f"Expected a directory: {directory}")
    result = []
    for current, directories, files in os.walk(directory, followlinks=False):
        for name in directories:
            no_symlinks(Path(current) / name)
        for name in files:
            path = Path(current) / name
            regular_file(path)
            result.append(path)
    return sorted(result)


def run_cli(command: list[str]) -> None:
    # The README invokes the venv interpreter directly, without shell activation.
    sibling = Path(sys.executable).absolute().parent / "ms-hub"
    if sibling.is_file():
        command = [str(sibling), *command[1:]]
    print("+ " + shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def existing_matches(source: Path, destination: Path) -> bool:
    no_symlinks(destination)
    if not destination.exists():
        return False
    regular_file(destination)
    if source.stat().st_size != destination.stat().st_size or sha256(source) != sha256(destination):
        raise AssetError(f"Existing file differs; refusing to overwrite {destination}")
    return True


def install_file(source: Path, destination: Path) -> None:
    """Atomically install a staged file without replacing an existing path."""
    if existing_matches(source, destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    no_symlinks(destination)
    # Staging lives under the release root, so both links share a filesystem.
    # link() fails if another process creates the destination in the meantime.
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        existing_matches(source, destination)


def download(args: argparse.Namespace, root: Path, manifest: dict) -> None:
    pending = []
    for artifact in manifest["artifacts"]:
        destination = safe_path(root, artifact["destination"])
        if destination.exists():
            verify_file(destination, artifact)
            print(f"Already verified: {artifact['destination']}")
        else:
            pending.append(artifact)
    upstream = [] if args.skip_upstream else manifest.get("upstream", [])
    # Reject existing symlinks and special files before any network access.
    for item in upstream:
        directory = safe_path(root, item["destination"])
        if directory.exists():
            scan_files(directory)
    with tempfile.TemporaryDirectory(prefix=".hub-assets-", dir=root) as temporary:
        staging = Path(temporary)
        installs = []
        for repo_type, repo_id, revision in (
            ("model", args.model_repo, args.model_revision),
            ("dataset", args.data_repo, args.data_revision),
        ):
            entries = [item for item in pending if item["repo_type"] == repo_type]
            if not entries:
                continue
            directory = staging / repo_type
            directory.mkdir()
            run_cli(["ms-hub", "download", repo_id, *[item["path"] for item in entries],
                     "--repo-type", repo_type, "--revision", revision,
                     "--local-dir", str(directory)])
            for item in entries:
                source = safe_path(directory, item["path"])
                verify_file(source, item)
                installs.append((source, safe_path(root, item["destination"])))
        for index, item in enumerate(upstream):
            directory = staging / f"upstream-{index}"
            directory.mkdir()
            run_cli(["ms-hub", "download", item["repo_id"], "--repo-type", "model",
                     "--revision", item.get("revision", "master"),
                     "--local-dir", str(directory), "--include", *UPSTREAM_FILES])
            files = scan_files(directory)
            runtime_files = [path for path in files if any(
                path.relative_to(directory).as_posix() == pattern or
                (pattern == "*.safetensors" and path.suffix == ".safetensors")
                for pattern in UPSTREAM_FILES)]
            names = {path.relative_to(directory).as_posix() for path in runtime_files}
            if "config.json" not in names or not any(name.endswith(".safetensors") for name in names):
                raise AssetError(f"Incomplete upstream snapshot: {item['repo_id']}")
            for source in runtime_files:
                relative = source.relative_to(directory).as_posix()
                installs.append((source, safe_path(root, item["destination"] + "/" + relative)))
        # Validate every destination before installing any newly downloaded file.
        for source, destination in installs:
            existing_matches(source, destination)
        for source, destination in installs:
            install_file(source, destination)
    print("Download complete. Release artifacts passed size and SHA256 checks.")
    if upstream:
        print("Upstream snapshots are supplied by the official Qwen repositories; their hashes are not pinned in this release manifest.")


def publication_files(root: Path, manifest: dict, repo_type: str) -> list[str]:
    directory = safe_path(root, "publish/" + ("model" if repo_type == "model" else "data"))
    expected = {item["path"]: item for item in manifest["artifacts"] if item["repo_type"] == repo_type}
    files = scan_files(directory)
    relative_files = [path.relative_to(directory).as_posix() for path in files]
    for relative in relative_files:
        if relative not in expected and relative not in METADATA_FILES:
            raise AssetError(f"Unlisted publication file: {directory / relative}")
    for relative, artifact in expected.items():
        verify_file(safe_path(directory, relative), artifact)
    if "README.md" not in relative_files:
        raise AssetError(f"Missing repository card: {directory / 'README.md'}")
    return relative_files


def upload(args: argparse.Namespace, root: Path, manifest: dict) -> None:
    commands = []
    for repo_type, repo_id, revision, folder in (
        ("model", args.model_repo, args.model_revision, "model"),
        ("dataset", args.data_repo, args.data_revision, "data"),
    ):
        files = publication_files(root, manifest, repo_type)
        print(f"{repo_type}: {repo_id}; {len(files)} validated files from publish/{folder}")
        for name in files:
            print(f"  {name}")
        commands.append(["ms-hub", "create", repo_id, "--repo-type", repo_type,
                         "--visibility", "private", "--exist-ok"])
        commands.append(["ms-hub", "upload", repo_id, str(root / "publish" / folder),
                         "--repo-type", repo_type, "--revision", revision,
                         "--commit-message", "Upload reviewed TRACE release assets",
                         "--no-cache", "--include", *files])
    print("New repositories are private; existing repository visibility is unchanged.")
    if not args.execute or manifest["public_release_approved"] is not True:
        print("DRY RUN: no repositories created and no files uploaded.")
        print("Execution requires --execute AND public_release_approved=true after a rights review.")
        for command in commands:
            print("Planned: " + shlex.join(command))
        return
    for command in commands:
        run_cli(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "upload"):
        command = subparsers.add_parser(name)
        command.add_argument("--release-dir", type=Path, default=Path(__file__).absolute().parents[1])
        command.add_argument("--model-repo", required=True, help="Release model repository: owner/name")
        command.add_argument("--data-repo", required=True, help="Release dataset repository: owner/name")
        command.add_argument("--model-revision", default="master")
        command.add_argument("--data-revision", default="master")
        if name == "download":
            command.add_argument("--skip-upstream", action="store_true", help="Skip the official Qwen base models")
        else:
            command.add_argument("--execute", action="store_true", help="Upload only if the manifest also records release approval")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_repo_id(args.model_repo)
        validate_repo_id(args.data_repo)
        validate_revision(args.model_revision)
        validate_revision(args.data_revision)
        root = Path(os.path.abspath(args.release_dir))
        no_symlinks(root)
        manifest = read_manifest(root)
        if args.command == "download":
            download(args, root, manifest)
        else:
            upload(args, root, manifest)
    except (AssetError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
