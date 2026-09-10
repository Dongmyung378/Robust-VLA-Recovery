# LIBERO Demonstration Data Card

[한국어](data-card.ko.md) | [Project README](../README.md)

## Scope and source

This card covers the four official LIBERO demonstration files selected for policy training in
Robust VLA Recovery. The audit ran on 2026-09-09 against
`yifengzhu-hf/LIBERO-datasets` revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. The source file names, byte sizes, and SHA-256
values are pinned in `configs/data_audit.toml`.

Upstream references:

- [Official LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Pinned LIBERO dataset revision](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90)

## Audited inventory

| Task | Source scene | File size | Demos | Transitions | Steps min / median / max |
|---|---|---:|---:|---:|---:|
| `pick_place` | `LIVING_ROOM_SCENE1` | 692,219,541 B | 50 | 6,939 | 113 / 138 / 173 |
| `stack` | `KITCHEN_SCENE2` | 641,644,880 B | 50 | 6,415 | 103 / 126 / 191 |
| `open_drawer` | `KITCHEN_SCENE1` | 472,881,442 B | 50 | 4,736 | 67 / 89 / 177 |
| `shelf_place` | `STUDY_SCENE4` | 800,762,622 B | 50 | 8,055 | 132 / 160 / 211 |
| Total | 4 scenes | 2,607,508,485 B | 200 | 26,145 | 67 / task-specific / 211 |

Task counts are balanced at 50 demonstrations each. Every valid episode contains `(T, 7)` actions
and two `uint8` RGB arrays with shape `(T, 128, 128, 3)`. Required simulator state, robot state,
reward, terminal, end-effector, gripper, and joint arrays share the same leading length.

## Quality audit

All 200 episodes passed the structural exclusion rules. The audit found no duplicate initial-state
hashes and no task-count imbalance. The following five episodes exceeded the robust within-task
length threshold. Start, middle, and final frames from both cameras were sampled for each episode;
the review found task progress and no blank or frozen stream.

| Task | Episode | Steps | Decision |
|---|---|---:|---|
| `stack` | `demo_42` | 191 | Retain; structurally valid |
| `open_drawer` | `demo_0` | 177 | Retain; structurally valid |
| `open_drawer` | `demo_6` | 149 | Retain; structurally valid |
| `open_drawer` | `demo_16` | 150 | Retain; structurally valid |
| `open_drawer` | `demo_40` | 141 | Retain; structurally valid |

Length warnings do not prove that a trajectory is incorrect. The sampled review supports retention
but does not replace a frame-by-frame semantic check during format conversion.

## Split policy

Each task uses 40 training, 5 validation, and 5 test demonstrations, for totals of 160, 20, and 20.
The ordering hash uses seed `378`. The upstream files do not store generator seeds, so the current
fallback group combines the file scene identifier and SHA-256 of the episode's initial simulator
state. A group is assigned intact to one split.

The manifest records the missing source seed as `null`; it does not invent one from the episode
number. If a future source revision exposes seeds, the auditor will use the explicit seed together
with the scene key.

## Exclusion and warning rules

An episode is excluded when any of these conditions holds:

- A required action, image, state, reward, or terminal array is missing or unreadable.
- An array is empty, contains non-finite numeric values, or disagrees on transition length.
- Actions are not seven-dimensional or contain values outside `[-1, 1]`.
- Either RGB stream is not `uint8` with shape `(T, 128, 128, 3)`.
- `num_samples` disagrees with the action length or the initial simulator state is absent.

The entire source file is rejected when its byte size or SHA-256 differs from the pinned revision,
its BDDL identity is inconsistent, or the declared demonstration count and transition total do not
match the HDF5 contents.

A length outlier, false final `done`, or duplicate initial state is a warning rather than an
automatic exclusion. These cases require review because a strict automatic rule could remove valid
human demonstrations.

## Storage and licensing

The four local source files occupy about 2.43 GiB and are excluded from Git. Audit reports and split
manifests are also local outputs. The official LIBERO repository describes its demonstration
datasets as CC BY 4.0, while the Hugging Face dataset card currently declares Apache 2.0 metadata.
The upstream terms must be confirmed before redistributing any demonstration file.

## Known limitations

- The audit verifies provenance and data structure, not whether every demonstrated action is
  optimal.
- The five length outliers were checked at three timepoints, not frame by frame.
- Source seed provenance is unavailable in this upstream revision.
- The four tasks use four named scenes, so this subset is not a general scene-diversity benchmark.
