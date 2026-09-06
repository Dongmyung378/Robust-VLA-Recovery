"""Version-scoped LIBERO backend for the Day 4 task contract."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from robust_vla_recovery.envs.libero_smoke import sha256_file
from robust_vla_recovery.envs.task_adapter import TaskSpec


class LiberoTaskBackend:
    """Confine LeRobot 0.6.1 / hf-libero 0.1.4 private API access here."""

    def __init__(self, spec: TaskSpec, *, width: int = 256, height: int = 256):
        from libero.libero import benchmark
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        from lerobot.envs.libero import LiberoEnv

        self.spec = spec
        suite = benchmark.get_benchmark_dict()[spec.suite](task_order_index=0)
        names = suite.get_task_names()
        if spec.name not in names:
            raise ValueError(f"installed LIBERO lacks task {spec.name}")
        self.task_id = names.index(spec.name)
        self.bddl_path = Path(suite.get_task_bddl_file_path(self.task_id))
        parsed = robosuite_parse_problem(str(self.bddl_path))
        if parsed["goal_state"] != [list(spec.goal)]:
            raise ValueError(f"BDDL goal mismatch: {parsed['goal_state']} != {[list(spec.goal)]}")
        self.native_language = " ".join(parsed["language_instruction"])
        expected_language = spec.instructions[0].rstrip(".").lower()
        if self.native_language.lower() != expected_language:
            raise ValueError(f"BDDL language mismatch: {self.native_language}")
        self.bddl_sha256 = sha256_file(self.bddl_path)
        self.env = LiberoEnv(
            task_suite=suite, task_id=self.task_id, task_suite_name=spec.suite,
            episode_length=spec.horizon,
            camera_name=("agentview_image", "robot0_eye_in_hand_image"),
            obs_type="pixels_agent_pos", observation_width=width, observation_height=height,
            init_states=True, episode_index=0, num_steps_wait=spec.settle_steps,
            control_freq=spec.control_frequency, control_mode="relative", hard_reset=True,
        )
        self.init_state_count = len(self.env._init_states)

    def reset(self, *, seed: int, init_state_index: int) -> tuple[Any, dict]:
        import numpy as np
        import torch

        if not 0 <= init_state_index < self.init_state_count:
            raise ValueError(f"init state index must be in [0, {self.init_state_count})")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # LeRobot normally increments this on every reset; force the requested
        # initial state each time so reset(seed=S, index=I) means exactly that.
        self.env.init_state_id = init_state_index
        observation, info = self.env.reset(seed=seed)
        self._assert_finite(observation)
        return observation, {**info, "libero_task_id": self.task_id,
                             "native_language": self.native_language,
                             "bddl_sha256": self.bddl_sha256}

    @staticmethod
    def _assert_finite(observation: dict) -> None:
        import numpy as np

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif value is None or not np.isfinite(np.asarray(value)).all():
                raise RuntimeError("nonfinite or missing simulator observation")

        visit(observation)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        import numpy as np

        observation, reward, done, truncated, info = self.env.step(
            np.asarray(action, dtype=np.float32)
        )
        try:
            self._assert_finite(observation)
        except RuntimeError:
            info["fatal_failure"] = "nonfinite_observation"
        return observation, reward, done, truncated, info

    def target_released(self) -> bool:
        if not self.spec.target_object:
            return True  # Drawer success is its native joint predicate.
        domain = self.env._env.env
        obj = domain.get_object(self.spec.target_object)
        if obj is None:
            raise RuntimeError(f"missing target object: {self.spec.target_object}")
        return not bool(domain._check_grasp(domain.robots[0].gripper, obj))

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()
