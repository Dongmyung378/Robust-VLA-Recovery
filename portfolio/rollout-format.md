# Lossless Rollout Format

[한국어](rollout-format.ko.md) | [Project README](../README.md)

## Transition alignment

The project stores the two original LeRobot `uint8` RGB streams without resizing, flipping, or
video compression. Side-by-side H.264 files are viewing derivatives and are never used as training
data.

An episode with T actions contains T+1 observations. Observation zero is the state after reset and
ten settling steps. At 20 Hz, the alignment is:

```text
observation[0], language[0] -- action[0] --> observation[1], reward[0], success[0]
       t = 0 ns                                  t = 50,000,000 ns
observation[1], language[1] -- action[1] --> observation[2], reward[1], success[1]
```

Timestamps are deterministic simulation offsets, not wall-clock times. Action t uses observation
and language t; reward and success t describe the resulting observation t+1. No padding action is
added after the final observation.

## Directory contract

```text
outputs/day05/batch_<UTC timestamp>_<random suffix>/
  batch.json
  verification.json
  <task>_seed<S>_init<I>_lang<L>/
    episode.h5
    metadata.json
    preview.mp4
```

Episode IDs are unique within a batch. New collection runs create a new directory and never
overwrite an existing episode. `batch.json` records the complete task, seed, initial-state, and
instruction matrix before execution. Its status changes from `running` to `complete` or `failed`.
Errors are preserved and are not retried automatically.

## HDF5 schema

| Dataset | Shape | Type and meaning |
|---|---|---|
| `observations/image` | `(T+1, H, W, 3)` | Original agent-view `uint8` RGB |
| `observations/image2` | `(T+1, H, W, 3)` | Original wrist-view `uint8` RGB |
| `observations/proprio` | `(T+1, 34)` | Robot state as `float64` |
| `observations/language` | `(T+1,)` | UTF-8 task instruction |
| `observations/frame_index` | `(T+1,)` | Integers from 0 through T |
| `observations/timestamp_ns` | `(T+1,)` | Frame index multiplied by 50,000,000 |
| `transitions/action` | `(T, 7)` | Finite `float32` values in `[-1, 1]` |
| `transitions/reward` | `(T,)` | Finite `float64` reward |
| `transitions/step_index` | `(T,)` | Integers from 0 through T-1 |
| `transitions/terminated`, `truncated` | `(T,)` each | Boolean end signals |
| `transitions/is_success`, `native_success` | `(T,)` each | Latched and current native success |
| `transitions/stable_success`, `target_released` | `(T,)` each | Diagnostic stability signals |

The 34 proprioceptive values contain end-effector position, quaternion, rotation matrix, gripper
position and velocity, and seven joint positions and velocities. This order is repeated in episode
metadata.

## Provenance

Metadata and HDF5 attributes record the episode ID, task name and ID, LIBERO suite, seed, initial
state, instruction index, horizon, task configuration, BDDL and catalog hashes, dependency versions,
policy, image orientation, and timestamp convention.

The current collection contract supports only the explicit clean condition
`{enabled: false, template_id: clean-v1}`. Unsupported perturbations are rejected rather than
recorded as if they had run. These rollouts validate the logging path; they are not expert
demonstrations or an independent evaluation set.

## Integrity and interruption handling

The writer validates indices, camera presence, fixed image shape, finite robot state, finite reward,
bounded action, language, and Boolean flags before each append. It flushes every transition and does
not accumulate the full episode in memory.

An active file is named `episode.partial.h5`, and its metadata status is `incomplete`. A normal
finish requires an end signal and a full schema check before the file is renamed to `episode.h5`.
An exception or unfinished context changes the status to `aborted`. A sudden process or power loss
may leave an `incomplete` file, which the verifier rejects.

Verification covers:

- Completion markers and matching provenance in JSON and HDF5
- Full-file SHA-256 and capture-time RGB pixel hashes
- Dataset lengths, dtypes, indices, and timestamps
- A terminal signal only on the last transition
- Latched success equal to the cumulative OR of native success
- Final outcome consistency with stability and release signals
- Decompression of every RGB frame
- Exact agreement with the configured task, seed, scene, and language matrix

The hashes detect accidental corruption but do not authenticate an artifact when both the data and
its recorded hash have been deliberately replaced.

## Compression and storage budget

RGB datasets use gzip level 1, shuffle, Fletcher32, and one-frame chunks. Storage planning starts
from the uncompressed RGB lower bound:

```text
bytes_RGB = episodes x (horizon + 1) x cameras x height x width x 3
20 x 401 x 2 x 256 x 256 x 3 = 3,153,592,320 bytes, about 2.94 GiB
```

Collection requires twice this RGB estimate plus 100 MB of free space for other arrays, HDF5
overhead, and metadata. Compression ratios from static no-op scenes are not used as guarantees for
future perturbed policy rollouts.

## Commands

```bash
rvla-rollouts collect --dry-run
rvla-rollouts collect --config configs/rollout.toml
rvla-rollouts verify outputs/day05/<batch>/batch.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json \
  --video outputs/day05/<batch>/<episode>/preview.mp4
```

Reading and verification require only `requirements/data.txt`. MP4 export additionally uses
`imageio` and `imageio-ffmpeg` from the simulation environment.
