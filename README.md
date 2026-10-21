# TRACE

**Typed Reference-Aware Conversational Editing for Humanoid Motion Diffusion**

[arXiv](https://arxiv.org/abs/2610.21515) · [PDF](https://arxiv.org/pdf/2610.21515) · [Models](https://modelscope.cn/models/kangzejian/TRACE) · [Inference assets](https://modelscope.cn/datasets/kangzejian/TRACE-Inference-Assets) · [Training](docs/TRAINING.md) · [Citation](#citation)

TRACE generates and edits Unitree G1 humanoid motion from language. An LLM predicts duration and sparse motion events; a diffusion Transformer combines this time-aligned Plan with the original text to generate a continuous trajectory. Reference-aware editing changes selected motion components while preserving local joint rotations outside the editing scope.

![TRACE method](media/method.png)

Motion-derived events supervise the language planner during training. At inference, the planner predicts events from text without access to a target motion. [Method details](docs/PAPER_OVERVIEW.md) · [Figure PDF](media/method.pdf)

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

`text → LLM Plan + text embedding → diffusion with optional pose conditions → smoothing → heel/toe refinement → SPS + video`

Omit `--boundary free` to use the bundled [start](examples/poses/idle_start.sps) and [end](examples/poses/idle_end.sps) idle references by default. The next example supplies these references explicitly for the same walking prompt.

- `motion_final.sps`: final 20 FPS motion for downstream use.
- `motion_raw.sps`: diffusion output before postprocessing.
- `pose35_final.npy`, `motion_final_qpos.npz`: array exports.
- `motion_smoothed.mp4`, `preview.png`, `index.html`: visualization of the final motion; the video filename is retained for compatibility.

Useful options:

- `--boundary free`: use the same boundary-capable model without start/end constraints.
- `--boundary none`: use the original generator without boundary conditioning.
- `--contact-refine none`: disable contact refinement; smoothing remains enabled.
- `--device cuda:0 --render-device 0`: select the inference and rendering GPU.
- `--skip-render`: export motion without rendering.

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

### Make a jumping jack higher

First generate the source motion, then edit that motion with a higher-jump request:

```bash
bash run.sh \
  --text "Perform a jumping jack." \
  --output outputs/jumping_jack

.venv/bin/python tools/experimental_reference_edit.py \
  --source outputs/jumping_jack/motion_final.sps \
  --instruction "Can you jump higher?" \
  --text "A person performs a high jumping jack, jumping well above the ground, then returns to standing." \
  --body lower_body --active-span 8 35 --height-gain-cm 14 \
  --strength 0.65 --cfg 3 \
  --output outputs/jump_edit
```

This example reuses `generator_boundary.pt` for reference-guided editing with a revised caption and Plan. The numerical height target is supplied explicitly and applied during denoising; it is not inferred from the follow-up sentence. `--active-span 8 35` is the zero-based, half-open editing interval for this example. For other source clips, choose an interval containing take-off, flight, and landing while excluding the idle endpoints.

Outputs include `motion_edited.sps` and before/after videos. This jump-height mode preserves endpoint poses and upper-body local joint rotations; root height and leg joints can change. No post-smoothing or external IK is applied to the edited motion.

<details>
<summary>Optional: render a local before/after overlay</summary>

Render both motions together, with **Before in blue** and **After in translucent orange**:

```bash
.venv/bin/python tools/render_edit_overlay.py \
  --edit outputs/jump_edit --output outputs/jump_overlay \
  --priority before
```

The synchronized overlay is saved to `outputs/jump_overlay/jump_edit/overlay.mp4`. Overlapping pixels show Before preferentially, without shifting or retiming either motion.

</details>

This generator-based path is experimental. The separate `edit.py` entry point supports trained relative hand-height editing. See [editing examples and supported scopes](docs/REFERENCES_AND_EDITING.md) for both interfaces.

## Training and data

[Training guide](docs/TRAINING.md) covers data preparation, event supervision, language-planner training, diffusion training, boundary fine-tuning, and the independent editor. [Dataset guide](docs/DATA.md) summarizes data cleaning and explains how to match the filename-only BONES-SEED splits to locally obtained data. The repository provides the training recipe and configuration; standalone generator/editor training entry points are not yet included.

Obtain motion files and annotations from the dataset provider under its license. Split lists contain filenames only, with no motion arrays, captions, or actor metadata.

The paper's split lists are included: [train](splits/bones_seed/train.jsonl) (84,287), [validation](splits/bones_seed/val.jsonl) (13,293), and [test](splits/bones_seed/test.jsonl) (13,302). See [split usage](splits/bones_seed/README.md) to match these filenames to your local data.

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
