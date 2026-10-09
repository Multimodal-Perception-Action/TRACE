# Release checklist

- [x] Owner approves publication; project-owned code and released artifacts use Apache-2.0.
- [x] Owner confirms public redistribution of derived weights and generated examples (2026-10-09); third-party training data remains excluded.
- [x] Data repository is limited to the inference resources listed in ARTIFACTS.md.
- [x] Repository targets: `Multimodal-Perception-Action/TRACE` on GitHub; `kangzejian/TRACE` and `kangzejian/TRACE-Inference-Assets` on ModelScope.
- [ ] Authenticate ModelScope locally; no access tokens in this folder.
- [ ] Pin and test exact upstream Qwen revisions.
- [x] Run offline unit tests, artifact hashing, and an end-to-end smoke test with authorized weights; see VALIDATION.md.
- [x] Review model and resource cards and retain third-party notices.
- [ ] Upload to private ModelScope repositories first, download into an empty directory, and check integrity and inference.
- [ ] Deliberately publish repositories after owner approval; tooling never silently changes visibility.

`release_manifest.json` records the owner's publication approval. Upload execution additionally requires `--execute`. This guardrail records a decision; it is not an independent legal assessment. ModelScope availability must be verified after upload, independently of local packaging.

`tools/release_check.py --archive` creates a code-only ZIP in `dist/`. It excludes weights, staged data, local output reports, private Git history, caches, environments, and credentials. Public examples contain selected videos and previews, not raw request/environment reports containing local paths.
