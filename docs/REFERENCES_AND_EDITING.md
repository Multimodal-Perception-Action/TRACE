# Pose references, checkpoint selection, and motion editing

## Independent start and end idle poses

Use `run.py` or `run.sh` with the boundary-conditioned generator:

```bash
bash run.sh \
  --text "A person walks forward at a steady pace." \
  --generator-checkpoint models/generator_boundary.pt \
  --generation-stats models/generation_stats.npz \
  --start-ref examples/poses/idle_start.sps --start-frame 0 \
  --end-ref examples/poses/idle_end.sps --end-frame 0 \
  --output outputs/walk_idle
```

Each reference is optional; an omitted side uses the corresponding bundled SPS
in `examples/poses/`.
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

The repository includes ready-to-use [start](../examples/poses/idle_start.sps)
and [end](../examples/poses/idle_end.sps) SPS references. Both contain the same
single-frame idle posture. They can be replaced independently, and need no
additional download. They are also the default references when no pose arguments
are supplied. Custom reference arguments do not change the model assets.

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

## Conversational editing

`converse.py` keeps a persistent session so a follow-up contains only the change:

```bash
bash run.sh --text "Briefly bow forward, then return to standing." \
  --contact-refine none --output outputs/bow
.venv/bin/python converse.py --source outputs/bow \
  --session outputs/bow_chat \
  --instruction "Bow more deeply, keeping the arm movements unchanged."
.venv/bin/python converse.py --session outputs/bow_chat \
  --instruction "Make the bow a little shallower."
```

The first call reads the caption from the generation directory's `request.json`.
For an external SPS without metadata, provide `--source-text` once. Later calls
must omit `--source` and reuse the same session. A compatible
`--generator-checkpoint` can be selected on the first call; weights and
normalization statistics are hashed and fixed for that session.

The data flow is:

```text
Latest motion + remembered caption + dialogue + new instruction
    → instruction-tuned Qwen base → typed relative intent + revised caption
    → measured source geometry + scope/target compiler
    → event-planner adapter + text embedding
    → source-noised diffusion + reference projection → updated motion
```

The instruction parser uses the shipped Qwen base without the event-only adapter.
The revised caption then goes through the existing event-planner adapter. These
are sequential inference stages, not newly trained model weights. The full
caption is now internal; users do not need to rewrite the action each turn.

Supported measured relative requests include jump height, kick/knee height,
crouch/lunge depth, and torso-bow angle. Duration, global travel, heading, and endpoint
poses remain fixed. Hand edits use the separate `edit.py` interface above.
Unsupported/ambiguous requests fail explicitly and do not advance the session.

For jump requests, FK measures the current source's lowest sole-point height
against the fixed ground. The compiler selects a unique jump or an apex with
at least twice the clearance of any competing jump in the requested region.
Comparable multiple jumps require a more specific time region. The edit window
covers take-off and landing without releasing the protected endpoint bands.
Jump edits release root height and both legs, preserving torso and arm rotations.
These are local-coordinate guarantees; world-space upper-body positions follow
the changed root height.

An unspecified jump magnitude uses a transparent policy: slight = 5 cm,
moderate = 10 cm, large = 14 cm. Direction is relative to the latest measured
source. Explicit distances replace the policy. Each change is limited to 25 cm;
the target sole clearance must stay within 2–45 cm and fit the unchanged duration.
These limits are inference guardrails, not physical-execution guarantees.
Numerical jump targets act inside denoising; no post-smoothing or external IK is
applied. Qualitative body-style edits use text/Plan conditions without a numerical
goal and may not realize the requested change.

### Source-relative crouch height

For an existing squat/crouch, `converse.py` measures the source pelvis-height
trajectory. "Crouch more deeply" lowers the measured minimum; "make the squat
shallower" raises it. Unspecified magnitudes use 3 cm for slight, 5 cm for
moderate, and 8 cm for large requests. Explicit distances replace these defaults.
The changed height follows the source crouch envelope, with zero change at the
protected boundaries. Each new turn measures its actual incoming motion again.

```bash
.venv/bin/python converse.py --source outputs/squat \
  --session outputs/squat_chat --instruction "Crouch 5 cm deeper."
.venv/bin/python converse.py --session outputs/squat_chat \
  --instruction "Make the squat 3 cm shallower."
```

During late denoising, the sampler projects root height onto this relative
trajectory and optimizes leg coordinates using differentiable four-point sole
FK, contact height, penetration, joint-range excess and temporal-correction
losses. Root planar motion, orientation, waist, arms and endpoint bands stay
protected. Upper-body world positions follow the changed root height; their
local rotations stay fixed. No additional post-smoothing is applied.

This is inference-time kinematic optimization with the existing checkpoint,
not a newly trained squat editor or a dynamics controller. Changes are limited
to 15 cm, the target pelvis height to 30–100 cm, and shallower requests must
retain at least 3 cm of measured crouch depth. Sources without a detectable
crouch, or windows that clip its minimum, are rejected. `result.json` reports
the requested/achieved height and sole-reference error for checking feasibility.
An output exceeding 0.1 cm root-profile error or 2.5 cm per-coordinate sole
reference RMSE is rejected before the session advances. This is a kinematic
sanity check, not a guarantee of simulator or robot stability.

`session.json` points to the latest completed output. Each numbered turn retains
`context.json`, the raw language response, `intent.json`, `compiled_edit.json`,
and the motion/overlay files. A schema error permits one logged language retry;
it is never replaced by a hardcoded semantic answer. Generation failure leaves
the preceding reference unchanged. Rendering failure is recorded separately
after a valid motion has been committed. A session lock prevents concurrent
turns from accidentally editing the same stale reference.

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

`--height-gain-cm` adds an explicit jump-height target to experimental reference
resampling. It is inference-time numerical guidance, separate from the trained
hand editor.

The guidance detects a source flight apex with differentiable G1 sole FK,
constructs a higher clearance envelope, and optimizes clean predictions during
the last denoising steps. The objective combines sole clearance, a root-height
flight target, stance contact, penetration, proximity to denoiser predictions,
and temporal continuity. Only root height and the two leg chains can change.
Source root orientation, horizontal travel, waist, arms, and all frames outside
the active interval remain exact. Height targets are soft optimization goals;
achieved heights must be measured from the decoded motion.

For a source containing a jump:

```bash
.venv/bin/python tools/experimental_reference_edit.py \
  --source /path/to/source_jump.sps \
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

Measure sole clearance, root peak height, joint limits, and penetration on the
decoded source and edited motion using the same fixed ground plane. These
kinematic checks do not replace simulator or controller validation.

`tools/optimize_jump_demo.py` retains the sampling sweep and numerical-guidance
trials. `tools/review_jump_edit.py` renders original / previous / improved
motions with a shared camera and produces a synchronized comparison plus a
source-priority overlay. No post-edit smoothing, motion retiming, or render-time
trajectory correction is used.
### Measured relative edits

Follow-up requests are interpreted relative to the **latest exported source motion**,
not the previous requested target. The conversational interface supports the
following measured attributes:

| Request | Measurement | Edited coordinates |
| --- | --- | --- |
| Kick higher / lower | Selected foot's mean sole-point height | Selected leg |
| Lift the knee more / less | Selected knee center height | Selected leg |
| Crouch or lunge deeper / shallower | Minimum pelvis height | Pelvis height and both legs |
| Bow deeper / shallower | Heading-relative torso forward angle | Pelvis tilt and both legs |

The default slight, moderate and large changes are 3, 5 and 8 cm for leg/pelvis
height, or 3, 5 and 8 degrees for bowing. Explicit amounts override these defaults
within the validated range. Jump-height editing retains its separate flight-height
policy. Deeper crouches/lunges **lower** the pelvis; deeper bows **increase** the
forward angle. A tapered source-event envelope applies the signed change while
keeping endpoint poses and event timing fixed.

```bash
python converse.py --source outputs/kick --session outputs/kick_edit \
  --instruction "Kick higher with the right leg, keeping the arm movements unchanged."
python converse.py --session outputs/kick_edit \
  --instruction "Make it a little lower."
```

The diffusion sampler uses differentiable kinematic guidance and reference
projection. This is an inference-time optimization, not a newly trained editor.
It preserves protected **local joint rotations**; world-space upper-body positions
can change when the pelvis moves or tilts. Bowing includes leg compensation to
retain the sole reference and removes residual yaw from optimized tilt coordinates.
Length and angle targets are serialized separately as `amount_cm` and `amount_deg`.

Source validation rejects motions without the requested measurable event. Output
validation checks target direction, target error, penetration and, for bowing,
sole-reference error before advancing the session. A source that barely lifts its
knee cannot meaningfully demonstrate a lower-knee edit. Rejected requests retain
the original motion and their diagnostic record. These checks establish kinematic
edit behavior; they do not establish dynamic stability or real-robot feasibility.
