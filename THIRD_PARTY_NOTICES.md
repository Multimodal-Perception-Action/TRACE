# Third-party provenance and notices

TRACE project-owned source code, released checkpoints, and project-owned inference assets are licensed under [Apache-2.0](LICENSE). Third-party materials retain the following licenses and attribution.

- Unitree G1 robot meshes and model resources: retain `assets/unitree_g1/LICENSE` (BSD-3-Clause, Unitree Robotics).
- The adapted G1 mocap XML matches the GMR asset at revision `bb1bbe40774794fceb2a7c579a3464a28e68c844` of [YanjieZe/GMR](https://github.com/YanjieZe/GMR). Its MIT notice is retained at `LICENSES/GMR-MIT.txt` in addition to the Unitree notice. These assets are not relicensed under the project's Apache-2.0 license.
- Qwen3-4B-Instruct-2507 and Qwen3-Embedding-4B are obtained separately from official Qwen repositories under their upstream Apache-2.0 terms. Preserve their license and attribution notices when distributing a copy. Those base-model licenses do not establish redistribution permission for TRACE-trained adapters/checkpoints or third-party data.
- The bundled pip wheel contains its own license and vendored notices. Runtime libraries installed through pip retain their upstream terms.
- Training data includes Motion Data by [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement. No raw or processed training dataset is redistributed in this release.

The project license does not grant rights to the underlying training datasets; see [LICENSING.md](LICENSING.md).

Upstream references:

- [BONES-SEED Dataset License Agreement](https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
- [GMR](https://github.com/YanjieZe/GMR)
- [ModelScope Hub tooling](https://github.com/modelscope/modelscope_hub)
