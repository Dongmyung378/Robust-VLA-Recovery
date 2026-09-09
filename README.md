# Robust VLA Recovery

[한국어 README](README.ko.md) | [Research protocol](portfolio/research-protocol.md) | [Data card](portfolio/data-card.md) | [Rollout format](portfolio/rollout-format.md)

Robust VLA Recovery studies whether temporal failure detection and constrained recovery
actions can improve vision-language-action manipulation under distribution shift. The project
uses LIBERO tasks to connect policy inference, failure detection, recovery, and OOD evaluation
under one reproducible experiment contract.

## Research question

> Does temporal failure detection with constrained rule-based recovery improve final task
> success under perturbations?

The evaluation compares clean-data training, perturbation augmentation, detector-only logging,
and detector-triggered recovery. Task success remains the primary outcome, while false recovery,
harmful recovery, detection lead time, and control latency measure the cost of intervention.

## System design

```text
language + RGB + robot state
            |
            v
       VLA policy ------> action ------> LIBERO environment
                                                |
                                                v
                                  temporal failure detector
                                                |
                                                v
                                      recovery manager
                                                |
                                  revised action or abort
```

The detector will use a causal window of recent visual features, robot state, and actions. The
recovery manager is a rule-based state machine with `reobserve`, `backoff`, `regrasp`,
`retry_subtask`, and `abort` primitives.

## Task suite

| Key | LIBERO task | Main failure modes |
|---|---|---|
| `pick_place` | Put alphabet soup in a basket | Failed grasp, wrong object, drop |
| `stack` | Stack the front black bowl on the middle bowl | Misalignment, drop, stall |
| `open_drawer` | Open the cabinet's top drawer | Wrong handle, stall, collision |
| `shelf_place` | Put the middle book on a cabinet shelf | Failed grasp, drop, collision |

All tasks come from `libero_90`. Each rollout has a 400-step limit at 20 Hz. LIBERO's native
predicate determines official success; a separate ten-step stability signal supports diagnosis.

## Experiment groups

| Group | Training data | Detector | Recovery | Purpose |
|---|---|---|---|---|
| A `Base` | Clean | No | No | Establish the clean-policy floor |
| B `Augmentation` | Clean and perturbed | No | No | Measure robust training alone |
| C `Detector` | Clean and perturbed | Log only | No | Measure detection quality and false alarms |
| D `Recover` | Clean and perturbed | Yes | Yes | Measure the net contribution of recovery |

The main causal comparison is D against B because both use the same robust policy. Detailed
success rules, metrics, and scope constraints are in the
[research protocol](portfolio/research-protocol.md).

## Current implementation

- Deterministic LIBERO task adapters with replayable video and trajectory metadata
- Lossless HDF5 rollout storage with checksums and T/T+1 alignment checks
- A deterministic lightweight policy used only to validate the control path
- A 40-episode clean-condition baseline evaluation across four tasks
- An official demonstration auditor with pinned source hashes and leakage-safe splits

The lightweight policy is not trained and does not represent VLA quality. Training starts after
the official demonstrations are converted to the learning format.

## Official demonstration audit

The four selected official LIBERO files are pinned to Hugging Face revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. The current audit found 200 valid
demonstrations, 26,145 transitions, no exclusions, and equal task counts. Five long episodes were
structurally valid, and sampled start, middle, and final frames showed no blank or frozen camera
stream.

| Task | Demos | Transitions | Steps min / median / max | Train / val / test |
|---|---:|---:|---:|---:|
| `pick_place` | 50 | 6,939 | 113 / 138 / 173 | 40 / 5 / 5 |
| `stack` | 50 | 6,415 | 103 / 126 / 191 | 40 / 5 / 5 |
| `open_drawer` | 50 | 4,736 | 67 / 89 / 177 | 40 / 5 / 5 |
| `shelf_place` | 50 | 8,055 | 132 / 160 / 211 | 40 / 5 / 5 |

The upstream HDF5 files do not expose generator seeds. The deterministic split therefore uses
seed `378` and keeps identical initial-state hashes within one split. See the
[data card](portfolio/data-card.md) for source, quality, exclusion, and licensing details.

## Quick start

Core validation requires Python 3.12 and does not install the simulator:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-deps -e .
rvla-validate-config configs/default.toml
python -m unittest discover -s tests -v
```

LIBERO execution uses Ubuntu 24.04 through WSL2:

```bash
conda env create -f environment.yml
conda activate robust-vla-recovery
python -m pip install uv==0.12.9
uv pip sync requirements/simulation-linux.lock --torch-backend cu128 --no-build-isolation
uv pip install --no-deps -e .
python -m unittest discover -s tests -v
```

## Main commands

| Command | Function |
|---|---|
| `rvla-run-libero-smoke configs/tasks/libero_smoke.toml` | Run one offscreen LIBERO episode |
| `rvla-run-tasks --task all --seed 378` | Run all four task adapters |
| `rvla-rollouts collect --config configs/rollout.toml` | Collect lossless rollouts |
| `rvla-policy-rollout run --config configs/policy.toml` | Validate policy-to-simulator control |
| `rvla-baseline-eval run --config configs/baseline_eval.toml` | Run the clean-condition baseline |
| `rvla-demo-audit plan` | Show the pinned official data inventory |
| `rvla-demo-audit run` | Audit demonstrations and generate deterministic splits |
| `rvla-demo-audit verify outputs/day08/<run>/audit.json` | Recheck the audit and source hashes |

The four official HDF5 files belong under `data/libero_90/`. Both `data/` and `outputs/` are
local-only and excluded from Git.

## Repository layout

```text
configs/       Frozen run, task, evaluation, and data-audit settings
portfolio/     Public research protocol, data card, and storage specification
src/           Installable Python package
scripts/       Simulator contract checks
tests/         Unit and integrity tests
data/          Official demonstrations, local only
outputs/       Videos, rollouts, evaluations, and audit reports, local only
local_notes/   Daily records and internal decisions, local only
```

## Reproducibility and limits

New runs use seed `378`. Source revisions, data sizes, file hashes, task definitions, and split
counts are pinned in versioned configuration. Large data and execution evidence remain local so
the repository stays reviewable.

The project currently covers simulation only. It does not claim real-robot transfer, and the
current baseline result must not be interpreted as trained VLA performance.
