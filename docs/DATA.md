# BONES-SEED data preparation

## What the split lists contain

The included [train.jsonl](../splits/bones_seed/train.jsonl), [val.jsonl](../splits/bones_seed/val.jsonl), and [test.jsonl](../splits/bones_seed/test.jsonl) identify the motions in each split. Each row has exactly one field:

```json
{"filename": "example_motion__A001.sps"}
```

The example name above is illustrative. Filenames retain their extension and mirror suffix. The lists contain no motion data, captions, actor metadata, or absolute filesystem paths. Obtain motions and annotations separately from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) under the provider's license.

| Split | Original clips | Mirrored clips | Total clips |
| --- | ---: | ---: | ---: |
| Train | 42,156 | 42,131 | 84,287 |
| Validation | 6,650 | 6,643 | 13,293 |
| Test | 6,653 | 6,649 | 13,302 |

These are clip counts before training-window segmentation.

## Cleaning and preprocessing

1. Use the Unitree G1 joint order and consistent coordinates at 20 FPS. Remove invalid non-root translation fields; retain root translation and joint rotations.
2. Match motions and annotations by filename, including mirror suffixes. Check file completeness and alignment before building training windows.
3. Keep related clips, original/mirrored pairs, and exact clip-level caption groups in the same partition. Use the provided lists to reproduce the split.
4. Divide long clips into windows of at most 512 frames, retaining their source split. Compute normalization statistics from the training set only.

Available mirrored motions use their own annotations; missing mirrors are not synthesized and left/right text is not blindly swapped. General phrases in temporal annotations are not used as grouping keys. The published lists preserve the paper's split membership and row order.

## Match the lists to local motions

From the repository root, match the included lists to your local dataset:

```bash
python tools/dataset_splits.py resolve \
  --splits splits/bones_seed \
  --motion-root /path/to/seed/g1/sps_20fps \
  --output-dir data/resolved
```

The resolver searches recursively and writes the corresponding motion paths relative to `--motion-root`. Missing files, duplicate basenames, cross-split duplication, and mirror pairs crossing splits cause an error. Existing output directories are never overwritten.

For the provider's CSV archive, add `--motion-suffix .csv` and point `--motion-root` to its G1 CSV directory. Matching by stem only locates the file; it does not convert coordinates, resample, or reproduce the 20 FPS training representation.

`manifest.json` records counts and SHA-256 checksums. For training, join the lists to locally licensed motions and annotations, then follow [TRAINING.md](TRAINING.md).
