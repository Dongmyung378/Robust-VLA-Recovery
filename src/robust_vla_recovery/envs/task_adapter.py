"""Four-task contract, independent of LIBERO imports and GPU dependencies."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from robust_vla_recovery.config import KNOWN_TASKS

RECOVERABLE_FAILURES = frozenset(
    {"grasp_failed", "object_dropped", "wrong_object", "stalled", "collision"}
)
FATAL_FAILURES = frozenset({"unsafe_state", "unrecoverable_failure", "operator_abort"})


@dataclass(frozen=True)
class TaskSpec:
    key: str
    suite: str
    name: str
    target_object: str
    goal: tuple[str, ...]
    instructions: tuple[str, ...]
    horizon: int
    control_frequency: int
    settle_steps: int
    stability_steps: int


def load_task_catalog(path: str | Path) -> dict[str, TaskSpec]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != {"schema_version", "defaults", "tasks"} or raw["schema_version"] != 1:
        raise ValueError("catalog requires schema_version=1, defaults, tasks")
    defaults = raw["defaults"]
    required = {"suite", "horizon", "control_frequency", "settle_steps", "stability_steps"}
    if not isinstance(defaults, dict) or set(defaults) != required:
        raise ValueError("invalid catalog defaults")
    if defaults["suite"] != "libero_90":
        raise ValueError("the frozen four-task catalog uses libero_90")
    for key in required - {"suite"}:
        value = defaults[key]
        if type(value) is not int or not 1 <= value <= 1000:
            raise ValueError(f"defaults.{key} must be an integer in [1, 1000]")
    if defaults["stability_steps"] > defaults["horizon"]:
        raise ValueError("stability window exceeds horizon")
    if not isinstance(raw["tasks"], dict) or set(raw["tasks"]) != KNOWN_TASKS:
        raise ValueError("catalog must contain exactly the four charter tasks")
    result = {}
    for key, task in raw["tasks"].items():
        if not isinstance(task, dict) or set(task) != {
            "name", "target_object", "goal", "instructions"
        }:
            raise ValueError(f"invalid task fields for {key}")
        if not isinstance(task["name"], str) or not task["name"].strip():
            raise ValueError(f"invalid task name for {key}")
        target = task["target_object"]
        if not isinstance(target, str) or (not target and key != "open_drawer"):
            raise ValueError(f"invalid target_object for {key}")
        goal = task["goal"]
        if not isinstance(goal, list) or not all(isinstance(x, str) and x for x in goal):
            raise ValueError(f"invalid goal for {key}")
        expected_predicate = {"pick_place": "in", "stack": "on",
                              "open_drawer": "open", "shelf_place": "in"}[key]
        if len(goal) != (2 if key == "open_drawer" else 3) or goal[0] != expected_predicate:
            raise ValueError(f"incorrect goal predicate for {key}")
        if target and goal[1] != target:
            raise ValueError(f"goal target mismatch for {key}")
        instructions = task["instructions"]
        if (not isinstance(instructions, list) or len(instructions) < 5
                or not all(isinstance(x, str) and x.strip() for x in instructions)
                or len(set(instructions)) != len(instructions)):
            raise ValueError(f"{key} requires at least five unique non-empty instructions")
        result[key] = TaskSpec(
            key=key, **defaults, **{k: task[k] for k in ("name", "target_object")},
            goal=tuple(goal), instructions=tuple(instructions),
        )
    if len({spec.name for spec in result.values()}) != 4:
        raise ValueError("catalog task names must be unique")
    return result


class TaskBackend(Protocol):
    def reset(self, *, seed: int, init_state_index: int) -> tuple[Any, dict]: ...
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]: ...
    def target_released(self) -> bool: ...
    def render(self) -> Any: ...
    def close(self) -> None: ...


class TaskAdapter:
    """Gymnasium-style reset/step with explicit native and diagnostic success.

    Native success is latched at its first occurrence. The episode may continue
    within its original budget to diagnose 10 consecutive successful, released
    states. This diagnostic never changes the official success label.
    """

    def __init__(self, spec: TaskSpec, backend: TaskBackend, *, instruction_index: int = 0,
                 init_state_index: int = 0):
        if (type(instruction_index) is not int
                or not 0 <= instruction_index < len(spec.instructions)):
            raise ValueError("instruction_index out of range")
        if type(init_state_index) is not int or init_state_index < 0:
            raise ValueError("init_state_index must be a nonnegative integer")
        self.spec, self.backend = spec, backend
        self.init_state_index = init_state_index
        self.instruction_index = instruction_index
        self.task = spec.name
        self.task_description = spec.instructions[instruction_index]
        self.active = False
        self.closed = False
        self.steps = 0
        self.first_success_step: int | None = None
        self.streak = 0
        self.last_info: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    def reset(self, seed: int = 0, **kwargs: Any) -> tuple[Any, dict]:
        if kwargs:
            raise ValueError(f"unsupported reset options: {sorted(kwargs)}")
        if self.closed:
            raise RuntimeError("create a new adapter after close")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2^32)")
        self.active = False
        self.steps, self.first_success_step, self.streak = 0, None, 0
        self.events = []
        obs, info = self.backend.reset(seed=seed, init_state_index=self.init_state_index)
        self.active = True
        self.last_info = {**info, "task_key": self.spec.key, "language": self.task_description,
                          "seed": seed, "init_state_index": self.init_state_index,
                          "is_success": False, "stable_success": False}
        return obs, self.last_info.copy()

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        if not self.active:
            raise RuntimeError("reset is required before step or after episode termination")
        try:
            values = list(action)
            valid = len(values) == 7 and all(
                math.isfinite(float(x)) and -1 <= float(x) <= 1 for x in values
            )
        except (TypeError, ValueError):
            valid = False
        if not valid or (hasattr(action, "shape") and tuple(action.shape) != (7,)):
            raise ValueError("action must be a finite 7-vector within [-1, 1]")
        try:
            obs, reward, backend_done, backend_truncated, info = self.backend.step(action)
            released = self.backend.target_released()
        except Exception:
            self.active = False
            self.events.append({"step": self.steps, "type": "simulator_error", "fatal": True})
            raise
        self.steps += 1
        native = bool(info["is_success"])
        if native and self.first_success_step is None:
            self.first_success_step = self.steps
        self.streak = self.streak + 1 if native and released else 0
        stable = self.streak >= self.spec.stability_steps
        fatal = info.get("fatal_failure")
        if not math.isfinite(float(reward)):
            fatal = "nonfinite_reward"
        if backend_done and not native:
            fatal = fatal or "backend_terminated"
        terminated = bool(fatal or stable)
        truncated = bool(not terminated and (backend_truncated or self.steps >= self.spec.horizon))
        reason = (str(fatal) if fatal else "stable_success" if stable else
                  "backend_truncated" if backend_truncated else "timeout" if truncated else None)
        self.last_info = {
            **info, "task_key": self.spec.key, "language": self.task_description,
            "native_success": native, "is_success": self.first_success_step is not None,
            "first_success_step": self.first_success_step,
            "task_time_seconds": (
                (self.first_success_step or self.steps) / self.spec.control_frequency
            ),
            "target_released": released, "success_streak": self.streak,
            "stable_success": stable, "step": self.steps, "termination_reason": reason,
        }
        if fatal:
            self.events.append({"step": self.steps, "type": str(fatal), "fatal": True})
        if terminated or truncated:
            self.active = False
        return obs, float(reward), terminated, truncated, self.last_info.copy()

    def report_failure(self, kind: str) -> None:
        """Accept external events; automatic failure detection belongs to Day 20."""
        if not self.active:
            raise RuntimeError("failure events require an active episode")
        if kind not in RECOVERABLE_FAILURES | FATAL_FAILURES:
            raise ValueError(f"unknown failure event: {kind}")
        fatal = kind in FATAL_FAILURES
        self.events.append({"step": self.steps, "type": kind, "fatal": fatal})
        if fatal:
            self.active = False
            self.last_info.update(termination_reason=kind, aborted=True)

    def render(self) -> Any:
        return self.backend.render()

    def close(self) -> None:
        if not self.closed:
            self.active, self.closed = False, True
            self.backend.close()
