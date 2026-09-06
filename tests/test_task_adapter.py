import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from robust_vla_recovery.envs.task_adapter import TaskAdapter, load_task_catalog
from robust_vla_recovery.envs.task_cli import main

CATALOG = Path(__file__).resolve().parents[1] / "configs/tasks/catalog.toml"
NOOP = [0, 0, 0, 0, 0, 0, -1]


class FakeBackend:
    def __init__(self):
        self.native = False
        self.released = True
        self.done = False
        self.truncated = False
        self.reward = 0.0
        self.calls = []
        self.close_count = 0
        self.error = False

    def reset(self, *, seed, init_state_index):
        self.calls.append((seed, init_state_index))
        return {"state": [seed, init_state_index]}, {}

    def step(self, action):
        if self.error:
            raise RuntimeError("backend failed")
        return {}, self.reward, self.done or self.native, self.truncated, {
            "is_success": self.native,
        }

    def target_released(self):
        return self.released

    def render(self):
        return "frame"

    def close(self):
        self.close_count += 1


class TaskAdapterTests(unittest.TestCase):
    def setUp(self):
        self.specs = load_task_catalog(CATALOG)
        self.backend = FakeBackend()
        self.adapter = TaskAdapter(self.specs["stack"], self.backend)

    def test_catalog_and_language_contract(self):
        self.assertEqual(set(self.specs), {"pick_place", "stack", "open_drawer", "shelf_place"})
        for spec in self.specs.values():
            self.assertEqual(len(set(spec.instructions)), 5)
            for index, language in enumerate(spec.instructions):
                adapter = TaskAdapter(spec, FakeBackend(), instruction_index=index)
                _, info = adapter.reset(seed=42)
                self.assertEqual(info["language"], language)

    def test_strict_catalog(self):
        original = CATALOG.read_text(encoding="utf-8")
        for old, new in [('horizon = 400', 'horizon = true'),
                         ('stability_steps = 10', 'stability_steps = 401'),
                         ('goal = ["on"', 'goal = ["in"'),
                         ('target_object = "alphabet_soup_1"', 'target_object = "wrong"')]:
            with self.subTest(new=new), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.toml"
                path.write_text(original.replace(old, new), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_task_catalog(path)

    def test_reset_preserves_requested_seed_and_initial_state(self):
        adapter = TaskAdapter(self.specs["pick_place"], self.backend, init_state_index=7)
        one = adapter.reset(seed=123)
        adapter.step(NOOP)
        two = adapter.reset(seed=123)
        self.assertEqual(one, two)
        self.assertEqual(self.backend.calls, [(123, 7), (123, 7)])
        self.assertEqual(adapter.steps, 0)

    def test_each_task_times_out_at_exact_horizon(self):
        for spec in self.specs.values():
            with self.subTest(task=spec.key):
                adapter = TaskAdapter(spec, FakeBackend())
                _, reset_info = adapter.reset()
                self.assertEqual(reset_info["seed"], 378)
                for index in range(spec.horizon):
                    _, _, terminated, truncated, info = adapter.step(NOOP)
                    self.assertFalse(terminated)
                    self.assertEqual(truncated, index == spec.horizon - 1)
                self.assertEqual(info["termination_reason"], "timeout")
                self.assertFalse(info["is_success"])
                with self.assertRaises(RuntimeError):
                    adapter.step(NOOP)

    def test_success_is_native_and_stability_requires_ten_consecutive_released_states(self):
        self.adapter.reset()
        self.backend.native = True
        self.backend.released = False
        _, _, terminated, _, info = self.adapter.step(NOOP)
        self.assertTrue(info["is_success"])
        self.assertFalse(terminated)
        self.assertEqual(info["first_success_step"], 1)
        self.backend.released = True
        for _ in range(5):
            self.adapter.step(NOOP)
        self.backend.native = False
        info = self.adapter.step(NOOP)[4]
        self.assertTrue(info["is_success"])
        self.assertEqual(info["success_streak"], 0)
        self.backend.native = True
        for index in range(10):
            _, _, terminated, truncated, info = self.adapter.step(NOOP)
            self.assertEqual(terminated, index == 9)
            self.assertFalse(truncated)
        self.assertTrue(info["stable_success"])
        self.assertEqual(info["task_time_seconds"], 1 / 20)

    def test_native_success_on_last_step_is_not_relabelled_failure(self):
        adapter = TaskAdapter(replace(self.specs["stack"], horizon=1), self.backend)
        adapter.reset()
        self.backend.native = True
        _, _, terminated, truncated, info = adapter.step(NOOP)
        self.assertTrue(info["is_success"])
        self.assertFalse(info["stable_success"])
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_stable_success_at_horizon_has_priority_over_timeout(self):
        adapter = TaskAdapter(replace(self.specs["stack"], horizon=10), self.backend)
        adapter.reset()
        self.backend.native = True
        for _ in range(10):
            result = adapter.step(NOOP)
        self.assertEqual(result[2:4], (True, False))

    def test_recoverable_events_continue_and_abort_prevents_steps(self):
        self.adapter.reset()
        for kind in ("grasp_failed", "object_dropped", "wrong_object", "stalled", "collision"):
            self.adapter.report_failure(kind)
            self.assertEqual(self.adapter.step(NOOP)[2:4], (False, False))
        self.adapter.report_failure("unsafe_state")
        with self.assertRaises(RuntimeError):
            self.adapter.step(NOOP)
        self.assertEqual(self.adapter.last_info["termination_reason"], "unsafe_state")
        self.assertEqual(len(self.adapter.events), 6)

    def test_invalid_action_rejected_without_advancing_simulator(self):
        self.adapter.reset()
        for action in ([0] * 6, [2] * 7, [float("nan")] * 7, [float("inf")] * 7, None):
            with self.subTest(action=action), self.assertRaises(ValueError):
                self.adapter.step(action)
        self.assertEqual(self.adapter.steps, 0)
        self.assertTrue(self.adapter.active)

    def test_backend_failure_terminates_and_preserves_event(self):
        self.adapter.reset()
        self.backend.error = True
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            self.adapter.step(NOOP)
        self.assertFalse(self.adapter.active)
        self.assertEqual(self.adapter.events[-1]["type"], "simulator_error")

    def test_backend_done_without_success_and_nonfinite_reward_are_fatal(self):
        for mode in ("done", "reward"):
            with self.subTest(mode=mode):
                backend = FakeBackend()
                setattr(backend, mode, True if mode == "done" else float("nan"))
                adapter = TaskAdapter(self.specs["stack"], backend)
                adapter.reset()
                self.assertEqual(adapter.step(NOOP)[2:4], (True, False))

    def test_lifecycle_and_bad_indices(self):
        with self.assertRaises(RuntimeError):
            self.adapter.step(NOOP)
        with self.assertRaises(ValueError):
            TaskAdapter(self.specs["stack"], self.backend, instruction_index=5)
        self.adapter.close()
        self.adapter.close()
        self.assertEqual(self.backend.close_count, 1)
        with self.assertRaises(RuntimeError):
            self.adapter.reset()

    def test_cli_rejects_unknown_task_before_loading_simulator(self):
        with self.assertRaises(SystemExit) as result:
            main(["--catalog", str(CATALOG), "--task", "bad-task"])
        self.assertEqual(result.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
