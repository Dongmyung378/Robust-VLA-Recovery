# Lossless rollout format v1

## Purpose and time alignment

Day 5 uses HDF5 as the authoritative training/debugging record. Both cameras are
stored losslessly as raw LeRobot `uint8` RGB, with no resize, flip or video codec.
The optional side-by-side H.264 MP4 is a viewing derivative only.

For an episode with T actions, there are T+1 observations. Observation 0 is the
state after reset and ten settling steps. At control frequency 20 Hz:

```text
observation[0], language[0] -- action[0] --> observation[1], reward[0], success[0]
       t = 0 ns                                  t = 50,000,000 ns
observation[1], language[1] -- action[1] --> observation[2], reward[1], success[1]
```

Timestamps are **simulation-relative** nanoseconds, not wall-clock inference
latency. They start after settling. Action t consumes observation/language t;
transition reward/success describe observation t+1. The language dataset has a
value at every observation even when the instruction stays constant. The final
observation has no following action; no padding action is invented.

## Directory layout

```text
outputs/day05/batch_<UTC timestamp>_<random suffix>/
  batch.json
  verification.json                    # Written by standalone batch replay
  <task>_seed<S>_init<I>_lang<L>/
    episode.h5                         # Published after terminal transition
    metadata.json                      # Completion marker + checksums
    preview.mp4                        # Optional derived visualization
```

Episode IDs are unique inside a batch; `(batch directory, episode_id)` is the
global identity. New collections get new batch directories and never overwrite
existing episodes. `batch.json` records the full task/seed/initial-state/language
matrix before collection. It stays `running` until all attempts finish, then
becomes `complete` or `failed`. Errors are retained without automatic retries.

## HDF5 datasets

| Dataset | Shape | dtype / content |
|---|---|---|
| `observations/image` | `(T+1, H, W, 3)` | uint8 raw agent camera RGB |
| `observations/image2` | `(T+1, H, W, 3)` | uint8 raw wrist camera RGB |
| `observations/proprio` | `(T+1, 34)` | float64 |
| `observations/language` | `(T+1,)` | UTF-8 text |
| `observations/frame_index` | `(T+1,)` | int64, exactly 0…T |
| `observations/timestamp_ns` | `(T+1,)` | int64, frame index × 50,000,000 |
| `transitions/action` | `(T, 7)` | float32, finite and in [-1, 1] |
| `transitions/reward` | `(T,)` | float64, finite |
| `transitions/step_index` | `(T,)` | int64, exactly 0…T−1 |
| `transitions/terminated`, `truncated` | `(T,)` each | bool |
| `transitions/is_success`, `native_success` | `(T,)` each | bool; latched vs current native success |
| `transitions/stable_success`, `target_released` | `(T,)` each | bool; Day 4 diagnostics |

Proprioception order: eef position (3), eef quaternion (4), eef orientation matrix
(9, row-major), gripper qpos (2), gripper qvel (2), joint position (7), joint
velocity (7). The ordered field description is also in metadata.

## Provenance

Metadata and HDF5 attributes contain matching provenance: episode ID, project
task key, native task name/ID/suite, seed, initial-state index, instruction index,
episode horizon, task specification, BDDL/catalog hashes, package versions,
policy identity, raw image orientation, and timestamp convention.

Day 5 explicitly records `perturbation = {enabled: false, template_id: clean-v1}`.
Unsupported perturbations are rejected; they must not be silently logged as if
applied. Perturbation generation is planned for Week 3. The current collection
is a storage smoke dataset using no-op actions; it is not expert demonstration
data or an independent evaluation split. Using five instructions does not test
language grounding because there is no language-conditioned policy yet.

## Integrity and interruption behavior

Each append validates the requested step index, both cameras, unchanged image
shapes, finite proprio/action/reward, action range, language and boolean flags
**before** writing. It writes one transition and the resulting observation and
flushes HDF5. Memory use does not grow with accumulated RGB frames.

While writing, the filename is `episode.partial.h5` and metadata is `incomplete`.
An exception or leaving the writer without `finish()` preserves the partial data
and marks metadata `aborted`. Finishing requires an end signal and validates the
whole T/T+1 structure before renaming to `episode.h5`. Metadata is replaced using
a temporary JSON file so a partial JSON write cannot certify completion.
Abrupt process/power termination can leave `incomplete` data; the verifier
rejects it. This is failure detection and evidence retention, not automatic
crash recovery or a promise of power-loss durability.

Verification checks:

- Complete markers and matching provenance in HDF5 and metadata.
- SHA-256 of the entire HDF5 file.
- All dataset lengths, dtypes, indices and timestamps.
- Terminal flags only at the final transition, with no fabricated terminal row.
- Latched success equals the cumulative OR of native success.
- Final outcome consistency and basic stable-success/release consistency.
- Decompression of every frame from both cameras. Replayed pixel hashes must
  match SHA-256 accumulated from the raw frames **at capture time**.
- The batch has exactly its configured task/seed/scene/language matrix; no
  omitted, duplicated or substituted episode is accepted.

Hashes detect accidental corruption, not deliberate rewriting of all artifacts
and hashes. Index/count checks detect logging omissions; they cannot determine
whether a simulator itself returned a stale image with an otherwise valid index.

## Compression and storage planning

RGB uses HDF5 gzip level 1, shuffle, Fletcher32, and one-frame chunks. This trades
larger files than temporal H.264 for lossless pixels, random frame access and
bounded-memory writes. Frame count and RGB dimensions predict the uncompressed
lower-level payload without relying on a compression-ratio guess:

```text
bytes_RGB = episodes × (horizon + 1) × cameras × height × width × 3
20 × 401 × 2 × 256 × 256 × 3 = 3,153,592,320 bytes ≈ 2.94 GiB
```

The collector checks for twice this RGB payload plus 100 MB of free space before
starting. Other arrays, HDF5 structure and metadata are additional overhead.
Compression depends on scene/noise; measured no-op episode sizes must not be
treated as guaranteed storage requirements for future perturbed policy rollouts.

## Commands

From repository root in the simulation environment:

```bash
python scripts/run_rollouts.py collect --dry-run
python scripts/run_rollouts.py collect --config configs/rollout.toml
python scripts/run_rollouts.py verify outputs/day05/<batch>/batch.json
python scripts/run_rollouts.py replay outputs/day05/<batch>/<episode>/metadata.json
python scripts/run_rollouts.py replay outputs/day05/<batch>/<episode>/metadata.json \
  --video outputs/day05/<batch>/<episode>/preview.mp4
```

Read/replay validation needs only `requirements/data.txt`; exporting MP4 also
requires imageio/imageio-ffmpeg from the simulation environment. `--dry-run` and
core imports work without data or simulation dependencies. GitHub CI includes a
data-only job so corruption tests do not silently rely on a GPU installation.
