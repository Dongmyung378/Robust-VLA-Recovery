# Policy Training Data Format

[한국어](training-format.ko.md) | [Project README](../README.md)

## Purpose

This format turns the audited LIBERO demonstrations into deterministic inputs for policy
fine-tuning. It preserves the original camera pixels and episode boundaries, records the declared
numeric casts, and makes the language-to-frame relationship explicit. The conversion reads one
episode at a time so the complete image corpus is never held in memory.

## Source and split

The converter rebuilds the Day 8 audit directly from the four files pinned in
`configs/data_audit.toml`. It rejects a changed byte size, SHA-256, dataset shape, or episode count
before writing training data. Split ordering uses seed `378`; initial-state groups remain intact
within one of the train, validation, or test partitions.

The converted set contains 200 episodes and 26,145 transitions. Each task contributes 40 training,
5 validation, and 5 test episodes.

## HDF5 layout

```text
dataset.h5
  episode_lengths        int64[200]
  episode_offsets        int64[201]
  episodes/
    000000/
      observation/
        agent_image      uint8[T, 128, 128, 3]
        wrist_image      uint8[T, 128, 128, 3]
        state            float32[T, 15]
        frame_index      int64[T]
        timestamp_ns     int64[T]
        language_index   int32[T]
      action             float32[T, 7]
      language_token_ids int32[L]
      language_attention_mask uint8[L]
```

The 15 state values concatenate `ee_states`, `gripper_states`, and `joint_states` in that order.
RGB arrays are copied byte for byte. State and action arrays are cast to `float32`. All large
arrays use lossless gzip level 1, shuffle, Fletcher32, and deterministic chunks.

`episode_offsets[i]` is the global starting transition for episode `i`, and the final offset equals
26,145. An episode group also records its task, source file, source episode, split, scene,
initial-state split group, instruction, and source seed availability.

## Temporal and language alignment

Frame `t` and action `t` form one policy-training row. `frame_index` contains integers from zero to
`T-1`, and `timestamp_ns[t]` equals `t * 50,000,000` at 20 Hz. Every `language_index` is zero,
which binds each frame in an episode to its single stored instruction and token sequence.

The transparent `lowercase-wordpunct-v1` tokenizer normalizes text with Unicode NFKC, lowercases
it, and separates words and punctuation. Its vocabulary is built only from training instructions,
with fixed `<pad>`, `<bos>`, `<eos>`, and `<unk>` identifiers. This tokenizer verifies data
alignment and vocabulary provenance. Day 10 may map the unchanged instruction text to the selected
policy's tokenizer.

## Validation results

Two complete conversions produced the same HDF5 and report checksums. The retained HDF5 is
1,192,042,683 bytes with SHA-256
`f8ec588217d3c5a19f350394d01f428d9380efd1ae1e2e74a53ca2d3ff082da0`.

The verifier reproduced all 200 split assignments from the pinned sources and checked episode
offsets, array shapes, dtypes, finite values, timestamps, language indices, tokens, metadata, and
the full-file checksum. A deterministic ten-sample review covered every task and all three splits.
Both camera arrays, the state vector, and the action array matched the corresponding source data.
The retained local contact sheet also showed identical source and converted frames for both
cameras.

## Commands

```bash
rvla-convert-demos plan
rvla-convert-demos run
rvla-convert-demos verify outputs/day09/<run>/conversion.json
rvla-convert-demos sample-check outputs/day09/<run>/conversion.json --count 10
rvla-convert-demos repro-check outputs/day09/<run-a>/conversion.json \
  outputs/day09/<run-b>/conversion.json
```

The source data, converted HDF5, manifests, and manual review image stay under ignored `data/` and
`outputs/` directories. Git contains only the conversion code, fixed configuration, tests, and
public format documentation.
