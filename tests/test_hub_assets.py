"""Offline safety and command-contract tests for the release asset helper."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "hub_assets", Path(__file__).absolute().parents[1] / "tools" / "hub_assets.py"
)
hub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hub)


class HubAssetsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payloads = {
            ("model", "generator.pt"): b"generator weights",
            ("model", "planner_adapter/adapter_config.json"): b'{"base": "Qwen"}',
            ("dataset", "generation_stats.npz"): b"generation statistics",
        }
        self.manifest = {
            "artifacts": [
                {
                    "repo_type": repo_type,
                    "path": name,
                    "destination": "models/" + name,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for (repo_type, name), payload in self.payloads.items()
            ],
            "upstream": [],
        }
        self.write_manifest()

    def write(self, relative, contents):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        return path

    def write_manifest(self):
        (self.root / "release_manifest.json").write_text(json.dumps(self.manifest))

    def invoke(self, command, *extra):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            status = hub.main([
                command, "--release-dir", str(self.root),
                "--model-repo", "test/release-model", "--data-repo", "test/release-data",
                *extra,
            ])
        return status, output.getvalue()

    def fake_download(self, command, check):
        self.assertTrue(check)
        self.assertEqual(command[:2], ["ms-hub", "download"])
        directory = Path(command[command.index("--local-dir") + 1])
        repo_type = command[command.index("--repo-type") + 1]
        for name in command[3:command.index("--repo-type")]:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.payloads[(repo_type, name)])
        # A CLI that also writes its own metadata cannot contaminate runtime assets.
        (directory / "README.md").write_bytes(b"Must not install this card")

    def test_valid_manifest(self):
        self.assertEqual(hub.read_manifest(self.root), self.manifest)

    def test_invalid_repository_ids(self):
        for value in ("model", "a/b/c", "--bad/name", "a/../b", "a b/c", "a/b;touch", "a\\b/c"):
            with self.subTest(value=value), self.assertRaises(hub.AssetError):
                hub.validate_repo_id(value)

    def test_unsafe_paths_rejected(self):
        for value in ("../escape", "/absolute", "a//b", "a/./b", "a\\b", "*.pt", "-arg", "a/../../b", "a\nb"):
            with self.subTest(value=value), self.assertRaises(hub.AssetError):
                hub.relative_path(value)

    def test_unsafe_revisions_rejected(self):
        for value in ("--force", "../master", "main\n--token", "/main"):
            with self.subTest(value=value), self.assertRaises(hub.AssetError):
                hub.validate_revision(value)

    def test_destination_cannot_escape_models(self):
        self.manifest["artifacts"][0]["destination"] = "run.py"
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_example_prompts_download_to_data_only(self):
        payload = b'{"text": "walk forward"}\n'
        self.payloads[("dataset", "example_prompts.jsonl")] = payload
        self.manifest["artifacts"].append({
            "repo_type": "dataset", "path": "example_prompts.jsonl",
            "destination": "data/example_prompts.jsonl", "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        self.write_manifest()
        self.assertEqual(hub.read_manifest(self.root), self.manifest)
        with patch.object(hub.subprocess, "run", side_effect=self.fake_download):
            status, output = self.invoke("download")
        self.assertEqual(status, 0, output)
        self.assertEqual((self.root / "data/example_prompts.jsonl").read_bytes(), payload)
        self.assertFalse((self.root / "models/example_prompts.jsonl").exists())

    def test_example_destination_exception_is_narrow(self):
        for repo_type, name, destination in (
            ("model", "example_prompts.jsonl", "data/example_prompts.jsonl"),
            ("dataset", "other_prompts.jsonl", "data/other_prompts.jsonl"),
            ("dataset", "generation_stats.npz", "data/generation_stats.npz"),
            ("dataset", "example_prompts.jsonl", "data/other_prompts.jsonl"),
            ("dataset", "example_prompts.jsonl", "models/example_prompts.jsonl"),
        ):
            with self.subTest(repo_type=repo_type, name=name, destination=destination):
                self.manifest["artifacts"][0].update(
                    repo_type=repo_type, path=name, destination=destination
                )
                self.write_manifest()
                with self.assertRaises(hub.AssetError):
                    hub.read_manifest(self.root)

    def test_bad_digest_or_size_rejected(self):
        for key, value in (("sha256", "00"), ("sha256", "F" * 64), ("bytes", -1), ("bytes", True)):
            with self.subTest(key=key, value=value):
                original = self.manifest["artifacts"][0][key]
                self.manifest["artifacts"][0][key] = value
                self.write_manifest()
                with self.assertRaises(hub.AssetError):
                    hub.read_manifest(self.root)
                self.manifest["artifacts"][0][key] = original

    def test_duplicate_destinations_rejected(self):
        duplicate = dict(self.manifest["artifacts"][0], repo_type="dataset")
        self.manifest["artifacts"].append(duplicate)
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_overlapping_destinations_rejected(self):
        artifact = dict(self.manifest["artifacts"][0], path="planner_adapter", destination="models/planner_adapter")
        self.manifest["artifacts"].append(artifact)
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_unreviewed_upstream_rejected(self):
        self.manifest["upstream"] = [{"repo_id": "mirror/qwen", "destination": "models/planner_base"}]
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_upstream_destination_cannot_change(self):
        self.manifest["upstream"] = [{"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "destination": "models/elsewhere"}]
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_upstream_cannot_overlap_artifact_exactly(self):
        self.add_upstream()
        self.manifest["artifacts"].append(dict(
            self.manifest["artifacts"][0], path="planner_base", destination="models/planner_base"
        ))
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_malformed_repo_types_rejected(self):
        self.manifest["artifacts"][0]["repo_type"] = []
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_malformed_upstream_repo_id_rejected(self):
        self.manifest["upstream"] = [{"repo_id": []}]
        self.write_manifest()
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_symlink_manifest_rejected(self):
        manifest = self.root / "release_manifest.json"
        real = self.root / "real.json"
        manifest.rename(real)
        manifest.symlink_to(real)
        with self.assertRaises(hub.AssetError):
            hub.read_manifest(self.root)

    def test_download_exact_artifacts_and_revisions(self):
        with patch.object(hub.subprocess, "run", side_effect=self.fake_download) as cli:
            status, output = self.invoke("download", "--model-revision", "v1", "--data-revision", "v2")
        self.assertEqual(status, 0, output)
        self.assertEqual(cli.call_count, 2)
        for artifact in self.manifest["artifacts"]:
            self.assertEqual((self.root / artifact["destination"]).read_bytes(), self.payloads[(artifact["repo_type"], artifact["path"])])
        self.assertFalse((self.root / "models/README.md").exists())
        commands = [call.args[0] for call in cli.call_args_list]
        self.assertIn("v1", commands[0])
        self.assertIn("v2", commands[1])
        self.assertIn("planner_adapter/adapter_config.json", commands[0])
        self.assertNotIn("--include", commands[0])
        self.assertNotIn("--token", commands[0])
        self.assertEqual(list(self.root.glob(".hub-assets-*")), [])

    def test_valid_existing_artifacts_skip_network(self):
        for artifact in self.manifest["artifacts"]:
            self.write(artifact["destination"], self.payloads[(artifact["repo_type"], artifact["path"])])
        with patch.object(hub.subprocess, "run") as cli:
            status, output = self.invoke("download")
        self.assertEqual(status, 0, output)
        cli.assert_not_called()

    def test_different_existing_artifact_fails_before_network(self):
        self.write("models/generator.pt", b"existing user data")
        with patch.object(hub.subprocess, "run") as cli:
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        cli.assert_not_called()
        self.assertEqual((self.root / "models/generator.pt").read_bytes(), b"existing user data")

    def test_bad_download_digest_installs_nothing(self):
        self.payloads[("dataset", "generation_stats.npz")] = b"corrupt"
        with patch.object(hub.subprocess, "run", side_effect=self.fake_download):
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        self.assertFalse((self.root / "models/generator.pt").exists())

    def test_symlink_destination_rejected_before_network(self):
        outside = self.root / "elsewhere"
        outside.mkdir()
        (self.root / "models").symlink_to(outside, target_is_directory=True)
        with patch.object(hub.subprocess, "run") as cli:
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        cli.assert_not_called()

    def test_file_as_destination_parent_rejected_before_network(self):
        self.write("models/planner_adapter", b"not a directory")
        with patch.object(hub.subprocess, "run") as cli:
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        cli.assert_not_called()

    def test_only_download_is_exposed(self):
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            hub.build_parser().parse_args(["upload"])

    def test_missing_cli_explained(self):
        with patch.object(hub.subprocess, "run", side_effect=FileNotFoundError("ms-hub not installed")):
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        self.assertIn("ms-hub not installed", output)

    def test_cli_uses_executable_beside_active_python(self):
        interpreter = self.write(".venv/bin/python", b"mock interpreter")
        executable = self.write(".venv/bin/ms-hub", b"mock hub executable")
        command = ["ms-hub", "download", "test/model", "config.json"]
        with patch.object(hub.sys, "executable", str(interpreter)), \
                patch.object(hub.subprocess, "run") as cli, \
                redirect_stdout(io.StringIO()):
            hub.run_cli(command)
        cli.assert_called_once_with(
            [str(executable), "download", "test/model", "config.json"], check=True
        )
        self.assertEqual(command[0], "ms-hub")

    def test_cli_failure_installs_nothing(self):
        with patch.object(hub.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "ms-hub")):
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        self.assertFalse((self.root / "models").exists())

    def fake_with_upstream(self, command, check):
        if command[2].startswith("Qwen/"):
            self.assertEqual(command[2], "Qwen/Qwen3-4B-Instruct-2507")
            self.assertIn("--include", command)
            directory = Path(command[command.index("--local-dir") + 1])
            (directory / "config.json").write_bytes(b'{"model_type":"qwen3"}')
            (directory / "model.safetensors").write_bytes(b"official model")
            (directory / "tokenizer.json").write_bytes(b"{}")
            (directory / "README.md").write_bytes(b"Do not copy")
        else:
            self.fake_download(command, check)

    def add_upstream(self):
        self.manifest["upstream"] = [{
            "repo_id": "Qwen/Qwen3-4B-Instruct-2507",
            "destination": "models/planner_base", "revision": "master",
        }]
        self.write_manifest()

    def test_upstream_is_official_and_cards_are_not_installed(self):
        self.add_upstream()
        with patch.object(hub.subprocess, "run", side_effect=self.fake_with_upstream):
            status, output = self.invoke("download")
        self.assertEqual(status, 0, output)
        self.assertEqual((self.root / "models/planner_base/model.safetensors").read_bytes(), b"official model")
        self.assertFalse((self.root / "models/planner_base/README.md").exists())

    def test_upstream_licenses_and_notices_are_retained(self):
        self.add_upstream()
        legal_files = {
            name: ("Upstream terms: " + name).encode()
            for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICE.md")
        }

        def download_with_legal_files(command, check):
            self.fake_with_upstream(command, check)
            if command[2].startswith("Qwen/"):
                directory = Path(command[command.index("--local-dir") + 1])
                for name, content in legal_files.items():
                    self.assertIn(name, command[command.index("--include") + 1:])
                    (directory / name).write_bytes(content)

        with patch.object(hub.subprocess, "run", side_effect=download_with_legal_files):
            status, output = self.invoke("download")
        self.assertEqual(status, 0, output)
        for name, content in legal_files.items():
            self.assertEqual((self.root / "models/planner_base" / name).read_bytes(), content)
        self.assertFalse((self.root / "models/planner_base/README.md").exists())

    def test_skip_upstream(self):
        self.add_upstream()
        with patch.object(hub.subprocess, "run", side_effect=self.fake_download) as cli:
            status, output = self.invoke("download", "--skip-upstream")
        self.assertEqual(status, 0, output)
        self.assertEqual(cli.call_count, 2)

    def test_upstream_existing_difference_preserved_and_nothing_installed(self):
        self.add_upstream()
        self.write("models/planner_base/config.json", b"custom settings")
        with patch.object(hub.subprocess, "run", side_effect=self.fake_with_upstream):
            status, output = self.invoke("download")
        self.assertEqual(status, 1, output)
        self.assertEqual((self.root / "models/planner_base/config.json").read_bytes(), b"custom settings")
        self.assertFalse((self.root / "models/generator.pt").exists())

    def test_install_does_not_overwrite(self):
        source = self.write("source", b"new")
        destination = self.write("destination", b"old")
        with self.assertRaises(hub.AssetError):
            hub.install_file(source, destination)
        self.assertEqual(destination.read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
