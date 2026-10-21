<h1 align="center">TRACE</h1>

<p align="center">
  <strong>Typed Reference-Aware Conversational Editing<br>for Humanoid Motion Diffusion</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.21515"><img src="https://img.shields.io/badge/arXiv-2610.21515-B31B1B?style=flat-square&amp;logo=arxiv&amp;logoColor=white" alt="arXiv: 2610.21515" height="24"></a>
  <a href="https://modelscope.cn/models/kangzejian/TRACE"><img src="https://img.shields.io/badge/ModelScope-Models-2563EB?style=flat-square" alt="ModelScope models" height="24"></a>
  <a href="https://modelscope.cn/datasets/kangzejian/TRACE-Inference-Assets"><img src="https://img.shields.io/badge/Resources-Inference_Assets-0F766E?style=flat-square" alt="Inference assets" height="24"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-64748B?style=flat-square" alt="Apache 2.0 license" height="24"></a>
</p>

<p align="center">
  <a href="#installation">Get started</a> &nbsp;·&nbsp;
  <a href="#generate-motion">Generate</a> &nbsp;·&nbsp;
  <a href="#edit-motion">Edit</a> &nbsp;·&nbsp;
  <a href="docs/TRAINING.md">Train</a> &nbsp;·&nbsp;
  <a href="https://arxiv.org/pdf/2610.21515">Paper PDF</a> &nbsp;·&nbsp;
  <a href="#citation">Cite</a>
</p>

---

## Overview

TRACE generates and edits Unitree G1 humanoid motion from language. An LLM predicts duration and sparse motion events; a diffusion Transformer combines this time-aligned Plan with the original text to generate a continuous trajectory. Reference-aware editing changes selected motion components while preserving local joint rotations outside the editing scope.

<p align="center">
  <img src="media/method.png" alt="TRACE overview: motion-grounded event planning, conditioned diffusion, and reference-aware editing" width="100%">
</p>

Motion-derived events supervise the language planner during training. At inference, the planner predicts events from text without access to a target motion.

<p align="center">
  <a href="docs/PAPER_OVERVIEW.md">Method details</a> &nbsp;·&nbsp;
  <a href="media/method.pdf">High-resolution figure</a>
</p>

## Real-robot demonstrations

https://github.com/user-attachments/assets/0ab65824-a948-437c-aff4-fc65bd9d6bce

Jumping-jack editing, in-the-wild instructions, and dance examples. Physical execution uses a downstream controller. [Download MP4](media/real_robot_demo.mp4).

## Installation

Linux, Python 3.10–3.12, and an NVIDIA GPU are recommended. The default environment uses PyTorch 2.11.0 with CUDA 12.8 wheels.

```bash
git clone https://github.com/Multimodal-Perception-Action/TRACE.git
cd TRACE
TRACE_PYTHON=python3.10 bash install.sh
```

Use an absolute interpreter path if `python3.10` is not on `PATH`. The installer creates a virtual environment without changing system Python. See [installation and offline usage](docs/INSTALL.md) for driver, rendering, and CPU options.

## Download models

```bash
# Only needed for private or access-controlled repositories:
.venv/bin/ms-hub login

.venv/bin/python tools/hub_assets.py download \
  --model-repo kangzejian/TRACE \
  --data-repo kangzejian/TRACE-Inference-Assets
.venv/bin/python run.py --check-models
```

The downloader checks TRACE artifact sizes and SHA-256 hashes. It also obtains the language bases from the official `Qwen/Qwen3-4B-Instruct-2507` and `Qwen/Qwen3-Embedding-4B` repositories. Allow about 16 GB for those two bases, plus the TRACE weights and caches. [Model files and formats](docs/ARTIFACTS.md)

## Generate motion

### Walking without endpoint constraints

```bash
bash run.sh \
  --text "A person walks forward at a steady pace." \
  --boundary free \
  --output outputs/walk
```

The output directory must be new. This example uses the boundary-capable generator without start/end pose constraints. The pipeline is:

```text
Text → LLM Plan + text embedding → diffusion with optional pose conditions
     → smoothing + heel/toe refinement → SPS + video
```

Omit `--boundary free` to use the bundled [start](examples/poses/idle_start.sps) and [end](examples/poses/idle_end.sps) idle references by default. The next example supplies these references explicitly for the same walking prompt.

<details>
<summary><strong>Output files</strong></summary>

- `motion_final.sps`: final 20 FPS motion for downstream use.
- `motion_raw.sps`: diffusion output before postprocessing.
- `pose35_final.npy`, `motion_final_qpos.npz`: array exports.
- `motion_smoothed.mp4`, `preview.png`, `index.html`: visualization of the final motion; the video filename is retained for compatibility.

</details>

<details>
<summary><strong>Useful options</strong></summary>

- `--boundary free`: use the same boundary-capable model without start/end constraints.
- `--boundary none`: use the original generator without boundary conditioning.
- `--contact-refine none`: disable contact refinement; smoothing remains enabled.
- `--device cuda:0 --render-device 0`: select the inference and rendering GPU.
- `--skip-render`: export motion without rendering.

</details>

### Walking with start and end poses

Ready-to-use SPS references are included: [start pose](examples/poses/idle_start.sps) and [end pose](examples/poses/idle_end.sps). Both contain the same single-frame G1 idle pose; replace either file to use different endpoints.

```bash
bash run.sh \
  --text "A person walks forward at a steady pace." \
  --generator-checkpoint models/generator_boundary.pt \
  --start-ref examples/poses/idle_start.sps --start-frame 0 \
  --end-ref examples/poses/idle_end.sps --end-frame 0 \
  --output outputs/walk_idle
```

Start and end poses can come from different files. Compatible generation checkpoints can be selected independently of the references. [Reference formats and checkpoint compatibility](docs/REFERENCES_AND_EDITING.md)

## Edit motion

### Edit through follow-up instructions

First generate a bow using the bundled start and end idle poses:

```bash
bash run.sh \
  --text "Briefly bow forward, then return to standing." \
  --contact-refine none \
  --output outputs/bow
```

Then request **Bow more deeply**, without rewriting the original action:

```bash
.venv/bin/python converse.py \
  --source outputs/bow \
  --session outputs/bow_chat \
  --instruction "Bow more deeply, keeping the arm movements unchanged." \
  --render-device 0
```

For the next turn, provide **only the requested change** and the session location:

```bash
.venv/bin/python converse.py \
  --session outputs/bow_chat \
  --instruction "Make the bow a little shallower."
```

The session remembers the source description, dialogue, and latest completed motion. The language frontend produces the revised caption and typed edit; the compiler resolves its body/time scope and numerical target from the current motion. Every turn edits the **previous turn's output**, not the initial clip. No separate `--text`, `--body`, or frame interval is needed. You can also start a session from your own generated output directory; its caption is read automatically.

For bowing, the target is relative to the measured current torso forward angle: deeper increases it and shallower reduces it. Unspecified strengths use slight = 3°, moderate = 5°, and large = 8°; an explicit angle replaces this default. The interface also supports relative jump height, kick/knee height, and crouch/lunge depth. See [conversation behavior and supported scopes](docs/REFERENCES_AND_EDITING.md#conversational-editing).

Each completed turn saves `edit/motion_edited.sps`, its interpreted request, and an overlay video under `outputs/bow_chat/turns/`. **Blue is Before; translucent orange is After**, aligned in time and world coordinates. Add `--skip-render` to export motion only.

Bowing preserves endpoint poses, root translation and heading, and waist/arm local joint rotations. Pelvis tilt and leg coordinates adapt together to realize the angle change while retaining the sole reference. This uses reference-guided diffusion with in-sampling kinematic optimization, without an additional post-smoothing stage. Requests that fail source or output checks leave the current motion unchanged.

<details>
<summary><strong>Optional: render a local before/after overlay</strong></summary>

Render both motions together, with **Before in blue** and **After in translucent orange**:

```bash
.venv/bin/python tools/render_edit_overlay.py \
  --edit outputs/bow_chat/turns/001/edit --output outputs/bow_overlay \
  --priority before
```

The synchronized overlay is saved to `outputs/bow_overlay/edit/overlay.mp4`. Overlapping pixels show Before preferentially, without shifting or retiming either motion.

</details>

This generator-based path is experimental. The separate `edit.py` entry point supports trained relative hand-height editing. See [editing examples and supported scopes](docs/REFERENCES_AND_EDITING.md) for both interfaces.

## Training and data

[Training guide](docs/TRAINING.md) covers data preparation, event supervision, language-planner training, diffusion training, boundary fine-tuning, and the independent editor. [Dataset guide](docs/DATA.md) summarizes data cleaning and explains how to match the filename-only BONES-SEED splits to locally obtained data. The repository provides the training recipe and configuration; standalone generator/editor training entry points are not yet included.

Obtain motion files and annotations from the dataset provider under its license. Split lists contain filenames only, with no motion arrays, captions, or actor metadata.

The paper's filename-only split lists are included:

| Training | Validation | Test |
| :---: | :---: | :---: |
| [84,287 entries](splits/bones_seed/train.jsonl) | [13,293 entries](splits/bones_seed/val.jsonl) | [13,302 entries](splits/bones_seed/test.jsonl) |

See [split usage](splits/bones_seed/README.md) to match these filenames to your local data.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Citation

If you use TRACE in your research, please cite our paper. [Download BibTeX](CITATION.bib).

```bibtex
@article{kang2026trace,
  title={TRACE: Typed Reference-Aware Conversational Editing for Humanoid Motion Diffusion},
  author={Kang, Zejian and Luo, Yifu and Shen, Boyang and Li, Jingxuan and Chen, Zekai and Shi, Zekun and Xu, Yongxue and Hou, Zhenyu and Wu, Bingqian and Zhang, Shaolei and others},
  journal={arXiv preprint arXiv:2610.21515},
  year={2026}
}
```

## License

TRACE code and project-owned released artifacts use [Apache-2.0](LICENSE). Third-party components retain their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). BONES-SEED is provided by [Bones Studio](https://bones.studio/) under its dataset agreement and is not relicensed by TRACE.

The outputs are flat-ground kinematic references. Robot execution requires a separately validated tracking controller.
