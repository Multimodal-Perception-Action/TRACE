# Local candidate validation

Validation performed on 2026-10-09. This records observed local checks, not certification on all machines or permission to publish.

- ModelScope helper: real candidate manifest and both upload directories passed dry-run validation; no upload or repository creation was executed. Seven manifest artifacts totaling 525,101,065 bytes passed local SHA-256/size checks.
- All 76 offline tests passed after release preparation. They cover environment bootstrapping, Plan serialization, SPS round trips, smoothing, contact Jacobians, fixed-frame preservation, asset paths and hub transfer safety.
- An end-to-end smoke run used the new directory's code with the existing authorized local engineering model directory (`--models`). The candidate artifact hashes match those model files; this test did not download from ModelScope or prove upstream-revision compatibility on a fresh network installation.
- Prompt: `A person walks forward at a steady pace.` The run produced 100 frames at 20 FPS, using standing boundaries, smoothing and `heel_toe` refinement. Local wall time including rendering was approximately 55 seconds; this is a single smoke test, not an efficiency benchmark.
- Final SPS matched `pose35_final.npy` within `1e-5`; all 16 fixed boundary frames matched the pre-refinement motion exactly; final label shape was `(100,2,2)`.
- The render report referenced `motion_final.sps`, with a matching content hash. Thus the preview used the postprocessed result, despite the retained `motion_smoothed.mp4` filename.
- Additional public waving and jumping-jack demos were generated with the same end-to-end workflow. Their video lengths are 5.05 and 2.55 seconds. Both retained fixed boundary frames exactly through contact refinement. Selected MP4 and PNG files are in `media/`; raw diagnostic reports remain excluded.
- Original runtime files were copied without algorithm changes. Updated release setup and hub tools are separate. No existing private engineering repository files were modified by candidate preparation.

Excluded validation outputs remain local under `outputs/`. They are not packaged in the code archive because request and environment reports contain machine-specific paths.
