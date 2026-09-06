# Robust VLA Recovery

Failure-aware vision-language-action manipulation with temporal failure
detection and closed-loop recovery in LIBERO.

## Status

- Day 1: project scope and evaluation contract frozen.
- Day 2: repository skeleton, reproducible environment specification, shared
  runtime configuration, validation, smoke tests, and CI completed.
- Day 3: WSL2/CUDA simulation stack, deterministic LIBERO smoke episode,
  offscreen video capture, trajectory metadata, and replay verification completed.
- Day 4: four named task adapters, 20 instructions, shared success/timeout/failure
  rules, one CLI, and real-simulator reset and success checks completed.
- Day 5: lossless streaming rollout logger, 20 collected and independently
  verified episodes (16,040 RGB frames), corruption tests, and storage budget completed.
- Day 6: image-state-language baseline policy adapter, action scaling, latency
  logging, manual failure review, and a verified 100-step LIBERO rollout completed.

See [the project charter](docs/project_charter.md) for the research question,
task definitions, metrics, and scope-reduction rules.

New single-seed runs default to `378`. Historical results retain their original
seeds; `configs/rollout.toml` retains the five-seed Day 5 reference plan.

## Repository layout

```text
configs/        Runtime and task configs
src/            Installable Python package
scripts/        Real-simulator contract checks
tests/          Fast unit and smoke tests
docs/           Project decisions and daily evaluations
outputs/        Generated runs (ignored by Git)
```

## Core smoke test

Install the package before using the `rvla-*` commands. These entry points
replace the former `scripts/run_*.py`, replay, and config-validation wrappers.
Empty future-feature packages and directory markers have been removed; create
those directories when their implementation is needed.

The core package has no third-party runtime dependency. Python 3.12 is the
supported version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
rvla-validate-config configs/default.toml
python -m unittest discover -s tests -v
```

## Simulation environment

LIBERO integration is Linux-only in LeRobot 0.6.1. On this Windows workstation,
create the simulation environment inside WSL2 or another Linux host:

```bash
conda env create -f environment.yml
conda activate robust-vla-recovery
python -m pip install uv==0.12.9
uv pip sync requirements/simulation-linux.lock --torch-backend cu128 --no-build-isolation
uv pip install --no-deps -e .
rvla-validate-config configs/default.toml
python -m unittest discover -s tests -v
```

`requirements/simulation-linux.txt` contains the direct simulation requirements;
the generated `requirements/simulation-linux.lock` is the reproducible 155-package
Linux/CUDA 12.8 lock used for installation.

## Day 3 LIBERO smoke episode

Run one fixed-seed, fixed-initial-state episode with two cameras and EGL
offscreen rendering:

```bash
conda activate robust-vla-recovery
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  rvla-run-libero-smoke configs/tasks/libero_smoke.toml
```

The runner creates a unique directory under `outputs/day03/` containing
`episode.mp4`, `trajectory.npz`, `first_frame.png`, and `metadata.json`.
Artifacts are intentionally ignored by Git. Verify and decode a saved episode
without a display server with:

```bash
rvla-replay-episode outputs/day03/<run>/metadata.json
```

The replay command rejects hash mismatches, inconsistent trajectory lengths,
and incomplete video decoding. See [the Day 3 evaluation](docs/day03_evaluation.md)
for the verified reference run and interpretation.

## Four task adapters (Day 4)

From the repository root in the simulation environment:

```bash
rvla-run-tasks --list
rvla-run-tasks --task all --seed 378
rvla-run-tasks --task stack --seed 378 --instruction-index 2
```

`--task` accepts `pick_place`, `stack`, `open_drawer`, `shelf_place`, or `all`.
Each task has five meaning-preserving instructions (indices 0-4). The catalog at
`configs/tasks/catalog.toml` fixes the native task name, target object, success
predicate, 400-step horizon, and 20 Hz control rate. The CLI resolves native IDs
by name and rejects a mismatch with the installed BDDL goal or language.

Every run saves video, metadata and trajectory under `outputs/day04/`, followed
by automatic full video decoding and checksum verification. A unique
`summary_*.json` records all task outcomes; the CLI exits nonzero on an execution
or verification error. `--list` works without simulation dependencies.

To repeat the separate simulator contract checks:

```bash
python scripts/check_task_adapters.py
```

These check repeatable resets, invalid initial-state rejection, recoverable and
fatal event handling, and an explicitly injected open-drawer success fixture.
The fixture is a contract test, not a policy rollout. All regular Day 4 runs use
fixed no-op actions; learned policy integration remains Day 6.

Native success is recorded immediately. The adapter can continue within the
same 400-step budget to diagnose ten consecutive successful/released states.
`is_success`, `native_success`, `stable_success`, and `first_success_step` keep
official success and this diagnostic distinct. See [Day 4 evaluation and API
contract](docs/day04_evaluation.md) for exact semantics and the verified results.

## Lossless rollout logger (Day 5)

The formal logger saves both RGB cameras, proprioception, actions, per-timestep
language, reward, success and termination flags in streaming HDF5. It preserves
T actions and T+1 observations with exact simulation timestamps, plus task/seed/
initial-state/instruction/perturbation provenance. Images use lossless gzip;
an H.264 preview can be exported separately.

```bash
rvla-rollouts collect --dry-run
rvla-rollouts collect --config configs/rollout.toml
rvla-rollouts verify outputs/day05/<batch>/batch.json
rvla-rollouts replay outputs/day05/<batch>/<episode>/metadata.json \
  --video outputs/day05/<batch>/<episode>/preview.mp4
```

The default plan is four tasks × five seeds = twenty 400-step no-op episodes.
The verifier decompresses every RGB frame, compares capture-time pixel hashes,
checks lengths/time alignment, and rejects incomplete or corrupted episodes.
See [the format and interruption contract](docs/rollout_format.md) for schema,
failure handling, compression, and storage estimates.
The [Day 5 evaluation](docs/day05_evaluation.md) records the executed batch,
42 passing tests, and measured 1.19 GiB storage for twenty episodes.

Data-only tests (no simulator or GPU required):

```bash
python -m pip install -r requirements/data.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## Baseline policy integration Day 6

The Day 6 diagnostic policy connects both RGB cameras, robot state, and language
to a bounded 7-dimensional action. Its seeded weights are untrained, so the run
checks the inference path and timing rather than manipulation performance.

```bash
rvla-policy-rollout run --config configs/policy.toml --dry-run
rvla-policy-rollout run --config configs/policy.toml
rvla-policy-rollout review outputs/day06/<run>/metadata.json \
  --candidate stalled --frame 0 --frame 100 --note "Manual review note"
```

The seed 378 reference run records 100 continuous policy steps. See the
[Day 6 evaluation](docs/day06_evaluation.md) for measurements and limitations.
