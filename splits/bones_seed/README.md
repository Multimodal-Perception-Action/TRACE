# BONES-SEED split lists

These are the filename-only train, validation, and test lists used by TRACE. They preserve the paper's split membership and row order.

| Split | File | Clips |
| --- | --- | ---: |
| Train | [train.jsonl](train.jsonl) | 84,287 |
| Validation | [val.jsonl](val.jsonl) | 13,293 |
| Test | [test.jsonl](test.jsonl) | 13,302 |

Each line contains only a filename, including its extension and mirror suffix:

```json
{"filename": "example_motion__A001.sps"}
```

The example above is illustrative. Original and mirrored versions stay in the same split. Counts refer to clips before training-window segmentation. [manifest.json](manifest.json) records file checksums and counts.

## Use with local data

Obtain the motions and annotations separately from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) under the provider's license. This folder contains no motion arrays, captions, or actor metadata.

Run from the TRACE repository root:

```bash
python tools/dataset_splits.py resolve \
  --splits splits/bones_seed \
  --motion-root /path/to/seed/g1/sps_20fps \
  --output-dir data/resolved
```

The resolver checks missing files, ambiguous basenames, duplicate membership, and cross-split mirror families. For provider CSV files, add `--motion-suffix .csv`; filename matching does not perform motion conversion. See the [data guide](../../docs/DATA.md) for preprocessing and split-construction details.
