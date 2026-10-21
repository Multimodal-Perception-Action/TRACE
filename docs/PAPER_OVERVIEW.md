# Paper overview

## TRACE

**TRACE: Typed Reference-Aware Conversational Editing for Humanoid Motion Diffusion**

[arXiv:2610.21515](https://arxiv.org/abs/2610.21515) · [Paper PDF](https://arxiv.org/pdf/2610.21515)

TRACE stands for **Typed Reference-Aware Conversational Editing**. The accompanying manuscript studies text-conditioned motion generation and source-relative motion editing for the Unitree G1 humanoid. Its central idea is to connect language to measurable motion structure: typed support, root, and hand events guide generation, while typed editing requests specify which parts of an existing motion should change and which should remain fixed.

This repository provides **generation and source-relative hand-height editing inference**. Generation includes independent idle-boundary references, compatible checkpoint selection, smoothing, contact refinement, export, and rendering. Editing includes the trained scope-language adapter, the separate diffusion editor, source-event extraction, and reference projection. The [training guide](TRAINING.md) records the training recipe and configuration; standalone generator/editor training entry points are not yet included.

## Method at a glance

### 1. Motion-grounded event planning

A text instruction often leaves timing, foot support, and posture underspecified. During training, TRACE extracts event targets from paired motions using forward kinematics and root reconstruction. These targets describe left/right foot support, root vertical motion, root height, and left/right hand height. Ambiguous labels carry confidence weights rather than being treated as certain negatives.

A supervised language model learns to predict both duration and typed event intervals from text. At inference, its output is rasterized into a 16-channel numerical Plan aligned with four-frame motion patches. A separate frozen text encoder retains the original caption's semantics, including details outside the compact event vocabulary. The diffusion Transformer receives both caption features and projected Plan features. Test-time planning does not require a target motion.

### 2. Gravity-referenced motion diffusion

The generator predicts continuous, 40-dimensional G1 states at 20 FPS. These states combine planar root displacement, absolute root height, yaw increments, gravity-relative root tilt, and 29 joint angles. Height and tilt are predicted directly; only planar displacement and yaw are integrated over time.

The paper groups its training objectives into motion reconstruction, integrated-geometry consistency, and forward-kinematics-based contact consistency. The contact objectives penalize support-foot sliding, drift, height error, and penetration. Training support masks come from the clean motion; predicted Plan support probabilities serve as conditioning rather than replacing those masks.

The paper's method figure primarily depicts this generation pathway, including the construction of motion-derived training supervision. Its edit-loop annotation refers to the separate editor described below.

### 3. Reference-aware editing in the paper

The full research method uses a separately trained diffusion editor. A supported follow-up request is parsed into a typed change and compiled into a body--time mask and a goal relative to the current motion. The editor conditions on source-motion features, revised events, the mask, and the requested change. At each denoising step, reference projection restores protected encoded coordinates; the final step copies those coordinates exactly.

The controlled editing experiments focus on explicit, source-relative hand-height changes. Their scope covers the selected arm's kinematic chain during the requested interval, with the root and other protected coordinates retained. Each edited output becomes the reference for the next request. Whole-body jump refinement is illustrated qualitatively and should not be confused with the quantitatively evaluated hand-height protocol.

The `edit.py` entry point exposes this bounded hand-height editing pathway with separate editor weights and a trained scope-language adapter. It accepts explicit left/right relative-height requests on short G1 motions; it does not expose general whole-body jump refinement. Standing boundary constraints remain a separate generation capability. See [supported inputs and examples](REFERENCES_AND_EDITING.md).

## Paper coverage and release coverage

| Component | In this inference release |
| --- | --- |
| Text-to-duration and typed-event planning | Yes, with the required pretrained assets |
| Caption- and Plan-conditioned G1 diffusion | Yes |
| Standing start/end constraints, smoothing, and heel/toe contact refinement | Yes; deployment-oriented additions to the inference workflow |
| SPS/array export and rendered previews | Yes |
| Source-relative conversational editing | Yes, for the released short-clip hand-height editing task |
| Training recipe and configuration | Yes; see [TRAINING.md](TRAINING.md) |
| Standalone generator/editor trainers and complete benchmark reproduction | No |
| Training/test motion data, captions, splits, and supervision caches | No |
| Robot control policy or hardware deployment stack | No |

The default release pipeline is:

`text -> predicted duration/events + caption embedding -> standing-conditioned diffusion -> smoothing -> heel/toe refinement -> motion files and preview`

The heel/toe postprocessor estimates geometric contact labels with a Plan prior and optimizes root translation and leg joints over time. It is not a newly trained heel/toe predictor, and it is distinct from the paper's contact-consistency training objectives. Its preservation and refinement behavior should be assessed using the release's own reports and tests. Postprocessed demonstrations are not a reproduction of the paper's benchmark protocol.

## Evidence and limitations

The manuscript evaluates generation on processed BONES-SEED motions, including text--motion alignment, distributional measures, human ratings, and matched component ablations. It evaluates scope localization separately from multi-turn hand-height editing under shared scopes and reference projection. Comparisons of complete editors do not isolate the effect of event conditioning alone. See the paper for benchmark results.

Both the method and this package assume text-conditioned G1 motion under flat-ground kinematics. They do not provide scene perception, general-purpose conversational editing, or guarantees of dynamic balance. Rendered motions and exported trajectories are kinematic references, not evidence of safe physical execution. Real-robot use requires a separately validated controller and an appropriate safety process.

See the [main README](../README.md) for installation, pretrained assets, usage, and licensing boundaries.
