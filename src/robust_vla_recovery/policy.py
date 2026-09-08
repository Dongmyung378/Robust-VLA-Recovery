"""Day 6 policy adapter and diagnostic rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import tomllib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from robust_vla_recovery.data.rollout import (
    EpisodeWriter,
    export_video,
    observation_arrays,
    verify_episode,
    write_json,
)
from robust_vla_recovery.envs.libero_smoke import prepare_libero_config, sha256_file
from robust_vla_recovery.envs.task_adapter import (
    RECOVERABLE_FAILURES, TaskAdapter, TaskSpec, load_task_catalog,
)


@dataclass(frozen=True, slots=True)
class PolicyRolloutConfig:
    catalog: Path
    output_dir: Path
    task: str
    seed: int
    init_state_index: int
    instruction_index: int
    steps: int
    warmup_calls: int
    action_scale: tuple[float, ...]
    gzip_level: int


def safe_repo_path(value: Any, field: str) -> Path:
    if (not isinstance(value, str) or not value.strip() or Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or ".." in value.replace("\\", "/").split("/")):
        raise ValueError(f"{field} must be a safe repository-relative path")
    resolved = Path(value).resolve()
    root = Path.cwd().resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"{field} escapes repository or equals its root")
    return Path(value)


def load_policy_config(path: str | Path) -> tuple[PolicyRolloutConfig, TaskSpec]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    expected = {"schema_version", "catalog", "output_dir", "task", "seed",
                "init_state_index", "instruction_index", "steps", "warmup_calls",
                "action_scale", "gzip_level"}
    if set(raw) != expected or raw["schema_version"] != 1:
        raise ValueError("invalid policy rollout schema")
    catalog_path = safe_repo_path(raw["catalog"], "catalog")
    output_dir = safe_repo_path(raw["output_dir"], "output_dir")
    catalog = load_task_catalog(catalog_path)
    task = raw["task"]
    if not isinstance(task, str) or task not in catalog:
        raise ValueError("unknown task")
    for key in ("seed", "init_state_index", "instruction_index", "steps", "warmup_calls",
                "gzip_level"):
        if type(raw[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    spec = catalog[task]
    if not 0 <= raw["seed"] < 2**32:
        raise ValueError("seed must be uint32")
    if raw["init_state_index"] < 0:
        raise ValueError("init_state_index must be nonnegative")
    if not 0 <= raw["instruction_index"] < len(spec.instructions):
        raise ValueError("instruction_index out of range")
    if not 100 <= raw["steps"] <= spec.horizon:
        raise ValueError("steps must be between 100 and the task horizon")
    if not 1 <= raw["warmup_calls"] <= 100:
        raise ValueError("warmup_calls must be in [1, 100]")
    scale = raw["action_scale"]
    if (not isinstance(scale, list) or len(scale) != 7
            or not all(type(value) in (int, float) and 0 < value <= 1 for value in scale)):
        raise ValueError("action_scale must contain seven numbers in (0, 1]")
    if not 0 <= raw["gzip_level"] <= 9:
        raise ValueError("gzip_level must be in [0, 9]")
    return PolicyRolloutConfig(
        catalog=catalog_path, output_dir=output_dir, task=task, seed=raw["seed"],
        init_state_index=raw["init_state_index"], instruction_index=raw["instruction_index"],
        steps=raw["steps"], warmup_calls=raw["warmup_calls"],
        action_scale=tuple(float(value) for value in scale), gzip_level=raw["gzip_level"],
    ), spec


class LightweightBaselinePolicy:
    """Small deterministic MLP for testing the complete policy I/O path.

    The weights are seeded but untrained. This is a systems baseline, not a
    manipulation-performance baseline.
    """

    IMAGE_FEATURES = 12
    STATE_FEATURES = 34
    LANGUAGE_FEATURES = 16
    HIDDEN_FEATURES = 32

    def __init__(self, *, seed: int, action_scale: tuple[float, ...]):
        import numpy as np

        if len(action_scale) != 7:
            raise ValueError("action_scale must have seven elements")
        self.np = np
        self.seed = seed
        self.action_scale = np.asarray(action_scale, dtype=np.float32)
        rng = np.random.default_rng(seed)
        inputs = self.IMAGE_FEATURES + self.STATE_FEATURES + self.LANGUAGE_FEATURES
        self.input_weights = rng.normal(0, inputs**-0.5,
                                        (inputs, self.HIDDEN_FEATURES)).astype(np.float32)
        self.output_weights = rng.normal(0, self.HIDDEN_FEATURES**-0.5,
                                         (self.HIDDEN_FEATURES, 7)).astype(np.float32)
        self.output_bias = np.asarray([0, 0, 0, 0, 0, 0, -2], dtype=np.float32)

    def encode(self, observation: dict, language: str):
        np = self.np
        pixels, state = observation_arrays(observation, language)
        image_features = []
        for camera in sorted(pixels):
            image = pixels[camera].astype(np.float32) / 255.0
            image_features.extend(image.mean(axis=(0, 1)))
            image_features.extend(image.std(axis=(0, 1)))
        digest = np.frombuffer(hashlib.sha256(language.encode("utf-8")).digest()[:16],
                               dtype=np.uint8).astype(np.float32)
        language_features = digest / 127.5 - 1.0
        return np.concatenate((np.asarray(image_features, dtype=np.float32),
                               np.tanh(state.astype(np.float32)), language_features))

    def predict(self, observation: dict, language: str):
        np = self.np
        hidden = np.tanh(self.encode(observation, language) @ self.input_weights)
        raw_action = np.tanh(hidden @ self.output_weights + self.output_bias)
        action = np.clip(raw_action * self.action_scale, -1.0, 1.0).astype(np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError("policy produced an invalid action")
        return action

    def contract(self) -> dict:
        digest = hashlib.sha256()
        for array in (self.input_weights, self.output_weights, self.output_bias):
            digest.update(array.tobytes())
        return {
            "name": "lightweight_mlp_v1",
            "role": "untrained deterministic systems baseline",
            "seed": self.seed,
            "weights_sha256": digest.hexdigest(),
            "inputs": {"cameras": ["image", "image2"], "proprioception": 34,
                       "language": "SHA-256 byte embedding"},
            "image_normalization": "uint8 RGB / 255 to [0, 1]; per-camera channel mean/std",
            "state_normalization": "elementwise tanh",
            "action_output": "tanh then elementwise scale and clip to [-1, 1]",
            "action_scale": self.action_scale.tolist(),
        }


def latency_summary(values_ns: list[int], control_frequency: int) -> dict:
    import numpy as np

    values_ms = np.asarray(values_ns, dtype=np.float64) / 1_000_000
    deadline = 1000 / control_frequency
    return {"calls": len(values_ns), "mean_ms": float(values_ms.mean()),
            "median_ms": float(np.median(values_ms)), "p95_ms": float(np.percentile(values_ms, 95)),
            "max_ms": float(values_ms.max()), "control_period_ms": deadline,
            "deadline_misses": int((values_ms > deadline).sum())}


def run_policy_rollout(config_path: str | Path) -> Path:
    config, spec = load_policy_config(config_path)
    if platform.system() != "Linux":
        raise ValueError("policy rollout requires Linux/WSL2")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    root = config.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prepare_libero_config(root)
    from robust_vla_recovery.envs.libero_tasks import LiberoTaskBackend

    backend = LiberoTaskBackend(spec)
    adapter = TaskAdapter(spec, backend, instruction_index=config.instruction_index,
                          init_state_index=config.init_state_index)
    policy = LightweightBaselinePolicy(seed=config.seed, action_scale=config.action_scale)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    episode_id = f"{stamp}_{config.task}_seed{config.seed}_policy"
    directory = root / episode_id
    latencies: list[int] = []
    actions = []
    primary_error: BaseException | None = None
    try:
        observation, _ = adapter.reset(seed=config.seed)
        for _ in range(config.warmup_calls):
            policy.predict(observation, adapter.task_description)
        policy_contract = policy.contract()
        provenance = {
            "episode_id": episode_id, "task_key": spec.key, "task_name": spec.name,
            "libero_task_id": backend.task_id, "suite": spec.suite, "seed": config.seed,
            "init_state_index": config.init_state_index,
            "instruction_index": config.instruction_index,
            "perturbation": {"enabled": False, "template_id": "clean-v1"},
            "horizon": config.steps, "control_frequency": spec.control_frequency,
            "policy": policy_contract, "catalog_sha256": sha256_file(config.catalog),
            "bddl_sha256": backend.bddl_sha256,
        }
        with EpisodeWriter(directory, provenance, observation, adapter.task_description,
                           control_frequency=spec.control_frequency,
                           gzip_level=config.gzip_level) as writer:
            for step in range(config.steps):
                started = time.perf_counter_ns()
                action = policy.predict(observation, adapter.task_description)
                latencies.append(time.perf_counter_ns() - started)
                actions.append(action.copy())
                observation, reward, terminated, truncated, info = adapter.step(action)
                if step == config.steps - 1 and not (terminated or truncated):
                    truncated = True
                    info = {**info, "termination_reason": "policy_rollout_limit"}
                writer.append(step, action, reward, observation, info["language"],
                              terminated=terminated, truncated=truncated, info=info)
                if terminated or truncated:
                    if step + 1 < config.steps:
                        raise RuntimeError(
                            f"episode ended after {step + 1} steps; need {config.steps}"
                        )
                    break
            metadata_path = writer.finish(final_info=info, events=adapter.events)
        verified = verify_episode(metadata_path)
        np = policy.np
        action_array = np.stack(actions)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["policy_inference"] = {
            **policy_contract, "warmup_calls_excluded": config.warmup_calls,
            "latency": latency_summary(latencies, spec.control_frequency),
            "scaled_action_min": action_array.min(axis=0).tolist(),
            "scaled_action_max": action_array.max(axis=0).tolist(),
            "continuous_steps": verified["steps"],
        }
        metadata["manual_failure_review"] = {"status": "pending"}
        write_json(metadata_path, metadata)
        verify_episode(metadata_path)
        export_video(metadata_path, directory / "preview.mp4")
        return metadata_path
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            adapter.close()
        except Exception as exc:
            if primary_error is None:
                raise
            primary_error.add_note(f"adapter cleanup also failed: {exc!r}")


def record_failure_review(metadata_path: str | Path, candidates: list[str], frames: list[int],
                          note: str) -> None:
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    verify_episode(path)
    invalid = set(candidates) - RECOVERABLE_FAILURES
    if not candidates or invalid:
        raise ValueError(f"invalid failure candidates: {sorted(invalid)}")
    steps = int(metadata["steps"])
    if not frames or any(type(frame) is not int or not 0 <= frame <= steps for frame in frames):
        raise ValueError("review frames must be within the episode")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("review note is required")
    metadata["manual_failure_review"] = {
        "status": "reviewed", "candidate_types": sorted(set(candidates)),
        "evidence_frames": sorted(set(frames)), "note": note.strip(),
        "candidate_only": True, "reviewed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(path, metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or review the Day 6 baseline policy")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/policy.toml"))
    run.add_argument("--dry-run", action="store_true")
    review = commands.add_parser("review")
    review.add_argument("metadata", type=Path)
    review.add_argument("--candidate", action="append", required=True,
                        choices=sorted(RECOVERABLE_FAILURES))
    review.add_argument("--frame", action="append", type=int, required=True)
    review.add_argument("--note", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config, spec = load_policy_config(args.config)
            if args.dry_run:
                print(json.dumps({**asdict(config), "catalog": str(config.catalog),
                                  "output_dir": str(config.output_dir),
                                  "control_frequency": spec.control_frequency}, indent=2))
            else:
                print(run_policy_rollout(args.config))
        else:
            record_failure_review(args.metadata, args.candidate, args.frame, args.note)
            print(f"Reviewed: {args.metadata}")
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"policy rollout error: {exc}\n")
    return 0
