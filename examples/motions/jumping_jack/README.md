# Jumping-jack reference

`source.sps` is the original generated starting reference for the paper's
qualitative jumping-jack demonstration. It contains 44 frames of G1 motion at
20 FPS. The displayed request is **Perform a jumping jack.**

Use it to reproduce the conversational editing example:

```bash
.venv/bin/python converse.py \
  --source examples/motions/jumping_jack/source.sps \
  --session outputs/jump_chat \
  --instruction "Can you jump higher?"
.venv/bin/python converse.py --session outputs/jump_chat \
  --instruction "A little lower, please."
```

The source is supplied as a fixed reference; subsequent edits are newly sampled
with the selected boundary-conditioned generator. Its existing endpoint bands
are preserved. This file does not replace the bundled idle pose references.
The original edited trajectory and robot recordings are not used as edit targets.

`request.json` supplies the source caption automatically. This is a generated
demonstration motion, not redistributed dataset ground truth or a test-set split
entry. All coordinates retain the original SPS convention.
