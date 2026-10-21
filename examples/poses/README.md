# Start and end pose examples

- [idle_start.sps](idle_start.sps): a single-frame Unitree G1 idle reference.
- [idle_end.sps](idle_end.sps): the same pose, provided separately so the end reference can be replaced independently.

The files contain root pose and 29 joint angles in TRACE's SPS format. Root translation is in centimeters, rotations are in degrees, and the frame index is `0`. Non-root translations are zero; the G1 skeleton supplies fixed joint offsets.

Run from the repository root after downloading the model assets:

```bash
bash run.sh \
  --text "A person stands and waves the right hand." \
  --generator-checkpoint models/generator_boundary.pt \
  --start-ref examples/poses/idle_start.sps --start-frame 0 \
  --end-ref examples/poses/idle_end.sps --end-frame 0 \
  --output outputs/custom_idle
```

These references are supplied with the code and are used by default; no additional pose download is needed. Explicit reference arguments replace the corresponding endpoint without modifying either bundled file. Compatible generator checkpoints can be selected independently of these files.

The constraints preserve posture in the beginning and ending idle bands, not an absolute world-space endpoint location or heading. See [pose references and editing](../../docs/REFERENCES_AND_EDITING.md) for formats and options.
