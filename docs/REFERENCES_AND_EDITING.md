# Pose references, checkpoint selection, and motion editing

## Independent start and end idle poses

Use `run.py` or `run.sh` with the boundary-conditioned generator:

```bash
bash run.sh \
  --text "A person stands and waves the right hand." \
  --generator-checkpoint models/generator_boundary.pt \
  --generation-stats models/generation_stats.npz \
  --start-ref my_start.sps --start-frame 0 \
  --end-ref my_end.sps --end-frame -1 \
  --output outputs/custom_idle
```

Each reference is optional; an omitted side uses its distributed default idle pose.
The files can be different, or can point to different frames of one recording.
References are copied into the new output directory as `boundary_reference_pose35.npy`;
the input files and default pose are never overwritten.

Accepted formats are named-joint G1 SPS and finite `pose35` NPY arrays of shape
`[35]` or `[T,35]`. Indices are zero-based; `-1` selects the last frame.
The references must already use the released G1 joint and coordinate conventions.
This interface switches **generation checkpoints**, not robot skeletons.

The first and last `--boundary-frames` frames are idle bands (default: eight frames).
Each pose is encoded independently, avoiding artificial velocity between different
reference poses. During denoising, known states are projected back to the corresponding
noise-level reference; the final projection restores them exactly. These constraints
specify root height, yaw-free tilt, joint pose, and zero local motion in each idle band.
They do not specify an absolute world-space endpoint position or heading.

The generation workflow protects the idle bands during smoothing and contact refinement.
Use `--contact-refine none` to inspect boundary conditioning with smoothing only.
`motion_raw.sps` is the diffusion output before smoothing or contact refinement.

For a self-contained example, create two synthetic references from the distributed idle:

```bash
.venv/bin/python tools/create_idle_examples.py
bash run.sh \
  --text "A person stands and waves the right hand." \
  --start-ref outputs/idle_examples/idle_start.sps \
  --end-ref outputs/idle_examples/idle_end.sps \
  --output outputs/two_idle_poses
```

The example end pose bends both elbows to 45 degrees to make the two references
visibly different. No private capture recording is used.

## Compatible generation checkpoints

`--generator-checkpoint` and `--generation-stats` select the denoiser export and
its matching normalization statistics. The LLM planner and text embedding remain
the released language interfaces.

| Mode | Export tag | Reference arguments |
| --- | --- | --- |
| `--boundary standing` | `boundary_events_v1` | Independent start/end poses |
| `--boundary free` | `boundary_events_v1` | No poses or endpoint projection; same boundary-capable model |
| `--boundary none` | `sparse_events_v1` | Not accepted |

Both formats require 40D G1 states, 2560D text features, four-frame patches,
128 maximum patches, the same sparse-event interface, and an EMA state dictionary.
Architecture settings are loaded from the export and its state dictionary is loaded
strictly. A raw training checkpoint or a different model family is not automatically
converted. If a model used different motion normalization, provide its own statistics.

```bash
bash run.sh --text "Briefly raise the right hand." \
  --boundary none --generator-checkpoint models/generator.pt \
  --generation-stats models/generation_stats.npz \
  --output outputs/original_generator
```

The chosen checkpoint filename, format, configuration and SHA-256 are recorded in
`checkpoint.json`; reference provenance is recorded separately. A mismatch fails
before diffusion instead of falling back to another model.

To disable endpoint control while retaining the boundary-capable weights, use
`--boundary free` without `--start-ref` or `--end-ref`. This sets every pose mask
entry to false and leaves no protected endpoint frames during smoothing or
contact refinement. `--boundary none` instead selects the original generator;
it is not a same-checkpoint boundary ablation.

## Source-relative editing

Editing uses the separate trained diffusion editor and edit-scope LoRA adapter:

```text
source motion + explicit edit instruction
  -> source summary + trained LLM edit parser
  -> typed relative change + body/time mask
  -> source events + revised event conditions
  -> source-conditioned diffusion with per-step reference projection
  -> edited SPS + before/after videos
```

Required editor-specific artifacts:

- `models/editor.pt`: event-aware relative-motion editor.
- `models/editing_stats.npz`: editor normalization, distinct from generator statistics.
- `models/edit_scope_adapter/`: trained scope parser, distinct from the generation adapter.

The parser shares the official Qwen base with generation; the edit instruction uses
the same embedding model but the editor-specific encoding instruction. The LLM,
embedding model, and editor run in separate processes to release memory between stages.

```bash
.venv/bin/python edit.py \
  --source my_motion.sps \
  --instruction "Raise the right hand by 8 cm during the middle half." \
  --output outputs/right_hand_edit
```

Supported source motions are 20 FPS G1 SPS or `pose35` arrays with 2–104 frames.
The nonzero transition interior needs at least three frames in the edit interval.
The released task supports one left or right hand-height change per call, up to
10 cm in either direction. Inputs outside this task are rejected. Long clips are
not automatically truncated or time-warped.

Explicit temporal expressions include `first half`, `middle half`, `last third`,
and `zero-based frames [20,60)`. The LLM predicts semantic fields, while explicit
time expressions are resolved by the same deterministic time compiler.
The offset is smoothly tapered with a squared-sine profile: the requested value
is the peak change, not a constant offset at every frame.

### Typed requests and restricted parser

To bypass language parsing, supply a JSON request:

```json
{
  "attribute": "hand_height",
  "effector": "right_hand",
  "delta_cm": 8,
  "time_span": [20, 60]
}
```

```bash
.venv/bin/python edit.py --source my_motion.sps \
  --instruction "Raise the right hand by 8 cm during the middle half." \
  --delta my_delta.json --output outputs/typed_edit
```

Alternatively, `--parser explicit` parses only the documented limited English
templates, such as `Raise the right hand by 8 cm during the middle half.` or
`Move the left hand 6 cm lower during the last half.` It does not run the LLM and
must not be presented as general language understanding.

### Outputs and preservation

- `motion_edited.sps`, `edited_pose35.npy`: edited motion.
- `source.sps`: the supplied source in the common export format.
- `delta.json`, `edit_mask.npy`, `requested_offsets_m.npy`: interpreted request.
- `revised_events.npy`: source-relative event conditions.
- `result.json`: checkpoint identity, preservation checks, and achieved palm-height diagnostics.
- `before/motion_smoothed.mp4`, `after/motion_smoothed.mp4`: synchronized-length
  replays with a shared camera; the filename is inherited from the renderer.

Create a side-by-side MP4 without changing playback speed:

```bash
.venv/bin/python tools/compare_edit.py --input outputs/right_hand_edit
```

This writes `comparison.mp4` and a midpoint preview `comparison.png`.

The root, legs, other arm and all out-of-scope joint coordinates are copied from
the source. Editing performs no post-smoothing or IK. Reference projection guarantees
protected-coordinate preservation, not exact task completion inside the editable mask.
The diagnostic reports achieved versus requested palm-height displacement and is
not a substitute for the paper's wrist-based benchmark.

Use the edited SPS as the next call's source for another supported request.
`--editor-checkpoint` and `--editing-stats` allow compatible editor weights;
the generation checkpoint argument does not change the editor.

## Additional qualitative editing trials

`tools/run_edit_showcase.py` runs six newly generated sources and ten actual
before/after trials. Both endpoints use the first frame of the supplied idle
reference; that reference is snapshotted inside the local output directory.

```bash
.venv/bin/python tools/run_edit_showcase.py --phase generate \
  --idle-ref my_idle.sps --output outputs/edit_showcase --gpus 0 1
.venv/bin/python tools/run_edit_showcase.py --phase edit \
  --idle-ref my_idle.sps --output outputs/edit_showcase --gpus 0 1
.venv/bin/python tools/compose_edit_showcase.py --input outputs/edit_showcase
```

The first four trials use the trained relative-hand editor: lowering either hand,
changing the time interval, and applying a second request to the preceding edited
motion. The remaining six are explicitly **exploratory generator-based reference
resampling**: deeper/shallower crouching, higher jumping, higher kicking, a deeper
bow, and switching the kicking side.

The exploratory path is implemented separately in
`tools/experimental_reference_edit.py`. It noises the source motion, uses a revised
full caption and LLM-predicted Plan, and restores protected coordinates during every
denoising step. Its body scope is supplied explicitly for each trial; it does not
claim automatic language-to-scope parsing or a trained whole-body editor. Target
Plan timing is mapped to the fixed source duration; the source motion is not warped.
Global travel and heading increments, plus the idle bands, remain protected.

Generation uses the release's smoothing stage. Neither editing path applies IK,
joint-trajectory corrections, or post-smoothing after editing. Both columns are
rendered with the same camera and playback rate. All completed trials remain in
the overview, including ineffective or wrong-direction edits. These videos show
kinematic outputs, not controller execution or demonstrated physical feasibility.

The compositor produces `all_edit_examples.mp4`, `trained_edit_examples.mp4`,
`exploratory_edit_examples.mp4`, and `overview.jpg`. It refuses to silently omit
missing cases. Per-case videos and SPS files remain in `edits/E01/` through
`edits/E10/`. No benchmark table is produced by the showcase.

### Superimposed before/after playback

Render both motions at their original coordinates and frame times:

```bash
.venv/bin/python tools/render_edit_overlay.py \
  --input outputs/edit_showcase --output outputs/edit_showcase/overlays --device 0
```

By default, the solid blue robot is the source, and the translucent orange robot
is the edited result. Both the three-quarter and side views show the two motions
superimposed. Source pixels take priority wherever the silhouettes overlap.
Exactly coincident edited surfaces are hidden to prevent z-fighting; neither
motion is translated, retimed, or postprocessed. Use `--priority after` to reverse
visibility priority.

The output contains `all_edit_overlays.mp4`, `overview.jpg`, and an `overlay.mp4`
plus a representative still for every case. Each still uses the frame with the
largest body-position difference; every original frame remains in the videos.
Use `--case E02 --still-only` with a separate output directory to preview styling.

To add six new source actions and fourteen actual editing trials, retaining the
original ten cases and idle reference:

```bash
.venv/bin/python tools/expand_edit_showcase.py \
  --previous outputs/edit_showcase --output outputs/edit_showcase_expanded --gpus 1 4
.venv/bin/python tools/render_edit_overlay.py \
  --input outputs/edit_showcase_expanded \
  --output outputs/edit_showcase_expanded/overlays --priority before
```

The additional sources include waving, punching, knee lifting, side leaning,
lunging, and rising onto the toes. Existing source and edit folders are referenced
without overwriting them. Failed jobs remain in the status files; completion of
inference is not a claim that every requested edit succeeded.

### Jump-height targets during exploratory resampling

Text-only reference resampling did not increase the jump in the W04 / WE08
celebration example. `--height-gain-cm` adds explicit, inference-time numerical
guidance to that experimental path. It is separate from the trained hand editor.

The guidance detects a source flight apex with differentiable G1 sole FK,
constructs a higher clearance envelope, and optimizes clean predictions during
the last denoising steps. The objective combines sole clearance, a root-height
flight target, stance contact, penetration, proximity to denoiser predictions,
and temporal continuity. Only root height and the two leg chains can change.
Source root orientation, horizontal travel, waist, arms, and all frames outside
the active interval remain exact. Height targets are soft optimization goals;
achieved heights must be measured from the decoded motion.

For the existing celebration source:

```bash
.venv/bin/python tools/experimental_reference_edit.py \
  --source outputs/wild_edits_20261020/edits/WE08/source.sps \
  --instruction "Jump higher while keeping the arm gesture." \
  --text "A person celebrates good news by raising both arms and jumping high above the ground." \
  --body lower_body --active-span 8 36 --height-gain-cm 14 \
  --strength 0.65 --cfg 3 --steps 50 \
  --output outputs/celebration_higher_jump
```

The interval is source-specific. Another motion needs its own jump detection and
an interval covering preparation, flight and landing. This option requires an
existing source jump and the boundary-compatible generator checkpoint. It does
not create a trained general-language jump editor.

The inspected `G14_v2` trial in `outputs/jump_edit_optimization_20261020/`
increased the minimum height of all eight sole points from 7.73 cm to 22.75 cm
at the apex. Root peak height increased from 90.49 cm to 100.27 cm. The preceding
unsuccessful edit reached 4.17 cm sole clearance. All measurements use the same
fixed ground plane; no per-motion height alignment is applied. The selected
output has no leg joint-limit violations, and its maximum sole penetration is
2.51 mm. These are kinematic checks, not simulator or controller validation.

`tools/optimize_jump_demo.py` retains the sampling sweep and numerical-guidance
trials. `tools/review_jump_edit.py` renders original / previous / improved
motions with a shared camera and produces a synchronized comparison plus a
source-priority overlay. No post-edit smoothing, motion retiming, or render-time
trajectory correction is used.
