# BONES-SEED V5 data splits

## What the split lists contain

The included [train.jsonl](../splits/bones_seed_v5/train.jsonl), [val.jsonl](../splits/bones_seed_v5/val.jsonl), and [test.jsonl](../splits/bones_seed_v5/test.jsonl) identify the motions in each split. Each row has exactly one field:

```json
{"filename": "example_motion__A001.sps"}
```

The example name above is illustrative. Filenames retain their extension and mirror suffix. The lists contain no motion data, captions, actor metadata, or absolute filesystem paths. Obtain motions and annotations separately from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) under the provider's license.

| Split | Original clips | Mirrored clips | Total clips |
| --- | ---: | ---: | ---: |
| Train | 42,156 | 42,131 | 84,287 |
| Validation | 6,650 | 6,643 | 13,293 |
| Test | 6,653 | 6,649 | 13,302 |

These are clip counts before training-window segmentation. The processed generator cache contains 86,385 training, 13,799 validation, and 13,706 test windows. Do not substitute window counts for original-clip evaluation counts.

## Split construction

V5 retains the prior data filtering and includes available original sagittal mirrors. Connected components join prior motion groups with mirror families and exact clip-level captions before assigning splits. Original/mirrored pairs remain in the same split. Missing mirrors are not synthesized. General phrases in temporal annotations are not used as grouping keys.

Temporal annotations are matched to the exact original or mirrored filename; left/right words are not blindly swapped. The filename-only export preserves the existing split membership and order rather than generating a new random split.

## Match the lists to local motions

From the repository root, match the included lists to your local dataset:

```bash
python tools/dataset_splits.py resolve \
  --splits splits/bones_seed_v5 \
  --motion-root /path/to/seed/g1/sps_20fps \
  --output-dir data/v5_resolved
```

The resolver searches recursively and writes the corresponding motion paths relative to `--motion-root`. Missing files, duplicate basenames, cross-split duplication, and mirror pairs crossing splits cause an error. Existing output directories are never overwritten.

For the provider's CSV archive, add `--motion-suffix .csv` and point `--motion-root` to its G1 CSV directory. Matching by stem only locates the file; it does not convert coordinates, resample, or reproduce the 20 FPS training representation.

## Export filename-only lists from existing V5 manifests

```bash
python tools/dataset_splits.py export \
  --source-dir /path/to/data_path_v5 \
  --output-dir data/v5_splits
```

Only the basename of `filepath` is retained. `manifest.json` records counts and SHA-256 checksums, with no source paths. The command does not copy or upload any dataset content.

For training, join the lists to locally licensed motions and annotations, then follow [TRAINING.md](TRAINING.md). Compute normalization statistics from the training split only.
