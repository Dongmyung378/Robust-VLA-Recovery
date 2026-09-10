# Research Protocol

[한국어](research-protocol.ko.md) | [Project README](../README.md)

## Objective

This project tests whether temporal failure detection and constrained rule-based recovery improve
vision-language-action manipulation under distribution shift. All comparisons use the same task
definitions, success predicates, rollout budget, and reporting rules.

## Hypotheses

- A robust policy with detector-triggered recovery will outperform the same policy without
  recovery under OOD conditions.
- A temporal detector will provide more useful pre-failure warning than a single-frame detector.
- Recovery will convert some failures into successes but may also introduce false or harmful
  interventions.

A negative result remains valid when the fixed protocol exposes the failure mode and intervention
cost clearly.

## Fixed scope

- Simulator: LIBERO tasks with LIBERO-Plus-style perturbations
- Preferred policy: SmolVLA
- Resource fallback: smaller SmolVLA settings, then ACT with a frozen CLIP text encoder
- Failure types: `grasp_failed`, `object_dropped`, `wrong_object`, `stalled`, `collision`
- Recovery actions: `reobserve`, `backoff`, `regrasp`, `retry_subtask`, `abort`
- Recovery states: `NORMAL`, `SUSPECT`, `RECOVER`, `VERIFY`, `ABORT`
- Deployment boundary: simulation only

## Tasks and success rules

| Task | Native goal | Additional diagnostic |
|---|---|---|
| `pick_place` | Alphabet soup is in the basket | Target is released for 10 consecutive steps |
| `stack` | Front black bowl is on the middle bowl | Target is released for 10 consecutive steps |
| `open_drawer` | Top drawer passes LIBERO's open threshold | Native success remains true for 10 steps |
| `shelf_place` | Middle book is in the upper shelf region | Target is released for 10 consecutive steps |

LIBERO's native predicate defines official task success. The ten-step signal diagnoses transient
contacts and does not replace the benchmark predicate. Every task uses a 400-step budget at 20 Hz.

## Comparison groups

| Group | Policy data | Detector | Recovery | Question |
|---|---|---|---|---|
| A `Base` | Clean | No | No | What is the clean-training floor? |
| B `Augmentation` | Clean and perturbed | No | No | How much does robust training add? |
| C `Detector` | Clean and perturbed | Log only | No | How accurate and costly are detector triggers? |
| D `Recover` | Clean and perturbed | Yes | Yes | What is the net contribution of recovery? |

The primary comparison is D minus B. Both groups must use the same policy checkpoint and evaluation
conditions so the difference isolates the detector and recovery system.

## Metrics

The primary outcomes are clean and OOD task success rates and the absolute OOD success-rate
difference between D and B. Secondary measures are:

- OOD degradation from the clean condition
- Failure-type Macro-F1 and AUROC
- Detection lead time relative to failure onset
- False triggers and false recovery on normal episodes
- Recovery rate and harmful recovery rate
- Task time, retry count, component latency, and total control latency
- Task-, perturbation-, severity-, and seed-level results with bootstrap 95% confidence intervals

## Data separation

Train, validation, and test partitions are grouped by source seed and initial scene state rather
than individual frames. Semantic paraphrases and perturbation templates must also remain disjoint
when they become part of the dataset. The official demonstration files audited for this project do
not expose generator seeds, so their split groups use the scene identifier and initial-state
SHA-256 as the available leakage boundary. Split ordering is deterministic with seed `378`.

Normal OOD evaluation, wrong-instruction stress tests, and training perturbations remain separate.
Failed or corrupt episodes are not silently regenerated; the exclusion or rerun reason must be
recorded.

## Resource constraints

The development machine has an NVIDIA RTX 3060 Laptop GPU with 6 GiB VRAM, about 64 GiB RAM, and
an Intel Core i7-11800H. A full training run must complete within 24 hours. RGB datasets are read
and written incrementally instead of being loaded into memory in full.

If the project exceeds its time or compute budget, scope is reduced in this order:

1. Remove `shelf_place` and retain three tasks.
2. Keep only camera, object, and language perturbations.
3. Reduce evaluation seeds from five to three per condition while keeping paired seeds.
4. Reduce image resolution, sequence length, and batch size; add gradient accumulation.
5. Replace SmolVLA with ACT and a frozen CLIP text encoder.
6. Complete binary failure prediction before multi-class detection.
7. Keep the rule-based recovery manager instead of adding a learned planner.

No reduction may remove the B-versus-D comparison, clean-versus-OOD reporting, or false and harmful
recovery measurements.

## Limits

This protocol does not cover physical-robot deployment, full fine-tuning of a 7B VLA, a learned
recovery planner, or model selection after inspecting final test outcomes. Success rules, test
seeds, and reporting metrics are frozen before the final evaluation stage.
