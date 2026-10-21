# Artifact contracts

## Model repository → local `models/`

| File | Purpose |
|---|---|
| `generator.pt` | Original sparse-event denoiser EMA and configuration |
| `generator_boundary.pt` | Endpoint-conditioned denoiser EMA and configuration |
| `planner_adapter/adapter_config.json` | PEFT LoRA configuration |
| `planner_adapter/adapter_model.safetensors` | Duration/event language planner adapter |
| `editor.pt` | Separate source-relative event-aware diffusion editor EMA |
| `edit_scope_adapter/adapter_config.json` | Edit-scope language adapter configuration |
| `edit_scope_adapter/adapter_model.safetensors` | Trained typed-request parser; separate from the duration/event planner |

These checkpoints are inference exports, not optimizer state. Use only trusted artifacts. PyTorch loading uses `weights_only=True` and strict state-dictionary matching. The two denoisers have different format tags; do not interchange them.

## Inference-resource dataset repository

- `generation_stats.npz` → `models/generation_stats.npz`: training-set mean/std for the 40D representation.
- `editing_stats.npz` → `models/editing_stats.npz`: the editor's own normalization; do not substitute generation statistics.
- `standing_pose35.npy` → `models/standing_pose35.npy`: two identical analytically constructed, FK-grounded standing poses with arms down. This is not a motion-capture clip.
- `example_prompts.jsonl` → `data/example_prompts.jsonl`: three newly authored generic prompts with no paired motion and no evaluation labels.

These inference assets do not contain training motions or captions. Filename-only training/validation/test lists are described separately in [DATA.md](DATA.md).

## Official upstream downloads

`models/planner_base/` comes from Qwen3-4B-Instruct-2507; `models/text_embedding/` comes from Qwen3-Embedding-4B. Their native tokenizer/config files must accompany all weight shards. The upstream revisions are recorded in `release_manifest.json`; `master` follows the upstream repository rather than an immutable snapshot.

## Contact outputs

In `heel_toe` mode, labels have shape `(T,2,2)`: frame, left/right foot, heel/toe. History adds an outer iteration dimension. In legacy `joint` mode labels are `(T,2)`. Joint optimization only changes root translation and leg joint angles; it is a CPU sparse solver even when generation and rendering use a GPU.

## Coordinate conventions

`pose35` stores SPS root translation in centimetres, root Euler xyz in degrees, then 29 G1 joint angles in degrees. MuJoCo exports reorder the root translation to xyz in metres, use a wxyz quaternion, and convert joints to radians. Consult `trace_core/motion_io.py` rather than inferring coordinates from array shape alone.
