# Training recipe

This guide records the training stages for TRACE's language planner, motion generator, optional boundary conditioning, and dedicated editor. Hyperparameters are also provided in [training_recipe.json](../configs/training_recipe.json). This JSON documents the recipe; it is not a configuration accepted by an executable training entry point in this repository. Standalone generator/editor trainers and the complete training-cache builder are not included yet.

## 1. Prepare motions and text

Obtain BONES-SEED motions and annotations from the provider and match them to the filename-only V5 lists as described in [DATA.md](DATA.md). The lists identify membership in train/validation/test; captions must be joined from the locally obtained annotations.

1. Use the Unitree G1 skeleton, consistent joint order, coordinate conventions, and 20 FPS sampling. Match each mirrored motion to its own annotation.
2. Segment long clips into windows of at most 512 frames without changing the source clip's split. Preserve the filename and window boundaries in the local cache.
3. Encode each window as gravity-gauged 40D states. Integrate planar displacement and yaw during decoding; root height and tilt are represented directly. The state and conversion implementation is in [trace_core/motion_io.py](../trace_core/motion_io.py).
4. Fit mean/standard deviation on training motions only and save them with the model. Do not reuse generator statistics for an independently trained editor.
5. Cache the original caption's frozen `Qwen/Qwen3-Embedding-4B` feature, with dimension 2,560. Keep motion, caption, event sequence, and window IDs aligned.

The paper cache contains 86,385 training windows. Window counts and the original-clip split sizes are different: evaluation must explicitly identify which unit it uses.

## 2. Derive event supervision from training motions

Forward kinematics provides support, root, and hand signals. A five-frame moving average precedes event quantization. Events are aligned to four-frame patches.

| Event group | Channels | Definition |
| --- | ---: | --- |
| Left/right stable support | 2 | Foot height and speed relative to the estimated ground |
| Root vertical movement | 3 | Descending / level / ascending; speed thresholds −0.08 and 0.08 m/s |
| Root height | 3 | Low / middle / high; height-above-ground thresholds 0.60 and 0.85 m |
| Left hand height | 4 | Height above the pelvis; thresholds 0.00, 0.25, and 0.45 m |
| Right hand height | 4 | Same height categories as the left hand |

Stable-support supervision uses a 0.035 m height threshold, a 0.2 m/s speed threshold, and a sequence ground estimate using the 0.05 quantile. Uncertain height/speed regions use margins of 0.025 m and 0.02 m/s. A missing stable-support label does not by itself mean airborne motion.

The language target serializes motion length and run-length event categories as JSON with `frames` and `events`. An event row contains its exclusive end-patch index followed by left support, right support, root vertical, root height, left hand, and right hand categories. Unknown categories use `-1`. See [llm_plan.py](../trace_core/llm_plan.py) for the schema and [llm_compile.py](../trace_core/llm_compile.py) for compilation. During conditioning, unknown support becomes 0.5 and unknown mutually exclusive categories become uniform distributions.

## 3. Train the text-to-Plan adapter

Fine-tune `Qwen/Qwen3-4B-Instruct-2507` on caption-to-event JSON pairs with supervised LoRA training. Targets are extracted from training motions; the model receives text, not the target motion, at inference.

- LoRA: all linear layers, rank 32, alpha 64, dropout 0.05; no new tokens, frozen token embeddings and language-model head.
- Two epochs; global batch 64, implemented with four GPUs, per-device batch 4, and gradient accumulation 4.
- Learning rate 5e-5, 100 warmup steps, cosine scheduling, BF16, gradient checkpointing, maximum sequence length 4,096.
- The training partition supplies 82,094 training examples and a 4,291-example inner holdout. The token audit found no overlength examples.
- Save the final adapter and its configuration. Parse and validate generated JSON before compiling the event sequence.

The inference code loads the adapter on its matching base model. Keep the base model identifier, adapter configuration, tokenizer, and serialization schema together when exporting a replacement.

## 4. Train the motion generator

### Text-conditioned initialization

Train the continuous generator from scratch for 32,000 optimizer updates using raw caption embeddings, global batch 256, learning rate 1e-4, 500 warmup steps, AdamW with weight decay 0.01, gradient clipping 1.0, and EMA 0.999.

The Transformer has eight layers, width 448, eight heads, and feed-forward width 1,792. Four-frame patches represent up to 512 frames. Training uses a 1,000-step cosine diffusion schedule, clean-state prediction, and condition dropout 0.1.

### Event-conditioned continuation

Initialize from the text model's EMA weights. Add a zero-initialized event projection and train for 8,000 updates with batch 128, learning rate 2e-5, 250 warmup steps, and EMA 0.999. Freeze the language planner and text encoder.

Each motion patch receives 16 event channels and four progress features through the event projection. Text and event residuals condition the denoiser; their contribution is attenuated in the final quarter of reverse denoising. See [events.py](../trace_core/events.py).

Training mixes motion-derived events and frozen LLM predictions with probability 0.5 for the motion-derived branch. Uncertain motion-derived entries retain the predicted condition. Training uses the true training-window length and rescales predicted event timelines by relative patch centers. Inference uses the text-predicted Plan without target-motion events.

The objective combines normalized clean-state reconstruction with the following weighted terms:

| Term | Weight |
| --- | ---: |
| Velocity / acceleration | 0.5 / 0.1 |
| Integrated planar position / yaw | 0.25 / 0.1 |
| Gravity consistency / yaw unit norm | 0.25 / 0.02 |
| Foot slip / foot anchor / contact height | 0.05 / 0.05 / 0.05 |
| Ground penetration, event-conditioned stage | 1.0 |

Ground-penetration weight is zero during the text-only initialization. The event-conditioned stage uses the lowest-sole L1 penetration objective with zero tolerance. Contact losses use ground-truth kinematic support, not predicted support labels. The configuration records contact thresholds, Huber scales, and anchor lags.

The released generator uses the fixed final 8,000-update event checkpoint. Default generation uses EMA weights, 50 DDIM steps, and CFG 2.5. Smoothing and heel/toe refinement are inference options, not part of the diffusion training target or raw-output benchmark.

## 5. Optional start/end conditioning

Continue from the event-conditioned generator for 4,000 updates, global batch 256, learning rate 2e-5, and EMA 0.999. Keep the planner, text encoder, event mixing, and base physics objectives unchanged.

Training supplies the actual ground-truth boundary fragments, with widths sampled from 1, 4, 8, 12, and 16 frames. Condition on both ends with probability 0.7, only the start with probability 0.2, and neither end with probability 0.1. Unknown reference entries are zeroed before projection. Add boundary MSE, seam-velocity MSE, and seam-acceleration MSE with weights 2.0, 1.0, and 0.1.

Training motions are not rewritten to start and end standing. At inference, an idle pose can replace the learned boundary inputs, and reference projection enforces known entries during denoising. This is a pose boundary condition, not an absolute world-position or heading destination. See [REFERENCES_AND_EDITING.md](REFERENCES_AND_EDITING.md).

## 6. Dedicated relative-motion editor

The dedicated editor is trained separately from the BONES-SEED generator. Retargeted MotionFix supplies 5,387 training and 330 validation pairs. Local supervision adds 3,690 pairs from 256 training sources using left/right hand edits, six temporal windows, signed 3/5/7 cm offsets, and instruction templates. Bounded IK constructs local training targets; it is not an inference-stage operation for this editor.

The editor uses independent weights/statistics, 40D states, four-frame patches, an eight-layer width-448 denoiser with eight heads, and a two-layer source encoder, for at most 104 frames. Scope and relative wrist offsets have separate zero-initialized projections. Offsets are normalized by 0.05 m.

| Stage | Updates | Main supervision / condition |
| --- | ---: | --- |
| Original editing | 8,000 | Retargeted source/instruction/target pairs |
| Mixed editing | 8,000 | Equal local and original pairs, source and scope |
| Relative goal | 4,000 | Relative offsets and wrist-goal losses |
| Event-aware continuation | 4,000 | Source-derived events and event consistency |

Original-edit training uses batch 192 and learning rate 1e-4. Relative-goal and event-aware stages use batch 128 with learning rates 5e-5 and 2e-5. The editor uses AdamW, EMA 0.99, a 300-step cosine diffusion schedule, and clean-state prediction.

Local losses combine reconstruction, velocity (0.2), and normalized wrist FK error (0.2). Goal losses weight vertical and horizontal wrist errors by 0.5 and 0.1, with squared distances normalized by `(0.05 m)^2`. The event-aware stage adds a zero-initialized projection of 16 source-event channels plus four progress features, and confidence-weighted hand-category cross entropy with weight 0.05. Only requested hand-height groups are revised using the relative offset.

Validation uses 64 original-validation and 40 expanded-development cases. Relative-goal and event-aware checkpoint selection compares steps 2,000 and 4,000 by mean joint success, with height RMSE as the tie-breaker; both select 4,000. Inference uses DDIM 50, CFG 1.5, and per-step reference projection without IK or post-smoothing. A separate instruction adapter maps supported text instructions to the editor request schema; inspect [edit_requests.py](../trace_core/edit_requests.py) and [edit_prompt.py](../trace_core/edit_prompt.py) when integrating a replacement parser.

## Export and validation

Export model configuration, EMA parameters, normalization statistics, and event/robot conventions together. Check compatibility before substituting a checkpoint; a renamed state dictionary is not a format conversion. [ARTIFACTS.md](ARTIFACTS.md) describes the inference artifacts.

Validate a small batch first: finite states, consistent dimensions, complete event coverage, frame counts, correct left/right conventions, and a rendered motion. Keep split membership unchanged and choose hyperparameters/checkpoints using training and validation data. Evaluate the chosen configuration on test separately.
