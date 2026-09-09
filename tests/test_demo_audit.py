import importlib.util
import tempfile
import unittest
from pathlib import Path

from robust_vla_recovery.data.demo_audit import (
    TASKS,
    _split_summary,
    assign_splits,
    inspect_demo_file,
    inspect_episode,
    load_audit_config,
    validate_split_manifest,
)
from robust_vla_recovery.data.rollout import file_hash

DATA_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "h5py"))
REPO = Path(__file__).resolve().parents[1]


class AuditConfigTests(unittest.TestCase):
    def test_config_pins_official_revision_files_and_default_seed(self):
        config, catalog = load_audit_config(REPO / "configs/data_audit.toml")
        self.assertEqual(config["split"]["seed"], 378)
        self.assertEqual(config["source"]["revision"],
                         "f13aa24a3da8c43c7225569f28c562979fa0e35a")
        self.assertEqual(set(config["tasks"]), set(TASKS))
        self.assertEqual(sum(config["split"][name]
                             for name in ("train", "validation", "test")), 50)
        for key, item in config["tasks"].items():
            self.assertEqual(Path(item["file"]).stem, f"{catalog[key].name}_demo")
            self.assertEqual(len(item["sha256"]), 64)

    def test_grouped_split_is_exact_deterministic_and_leak_free(self):
        episodes = []
        for index in range(6):
            group = "shared" if index < 2 else f"group-{index}"
            episodes.append({
                "episode_id": f"demo_{index}", "status": "valid", "steps": index + 1,
                "source_seed": None, "scene_id": "KITCHEN_SCENE2",
                "scene_key": f"KITCHEN_SCENE2:{group}", "split_group": group,
            })
        split = {"seed": 378, "train": 4, "validation": 1, "test": 1}
        first = assign_splits("stack", episodes, split)
        second = assign_splits("stack", list(reversed(episodes)), split)
        self.assertEqual(first, second)
        self.assertEqual(_split_summary(first)["stack"],
                         {"train": 4, "validation": 1, "test": 1})
        shared = {row["split"] for row in first if row["split_group"] == "shared"}
        self.assertEqual(len(shared), 1)

    def test_split_validation_rejects_group_leakage(self):
        rows = [
            {"task": "stack", "episode_id": "demo_0", "split": "train",
             "split_group": "same"},
            {"task": "stack", "episode_id": "demo_1", "split": "test",
             "split_group": "same"},
        ]
        manifest = {"schema_version": 1, "episodes": rows, "counts": _split_summary(rows)}
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_split_manifest(manifest)


@unittest.skipUnless(DATA_AVAILABLE, "install requirements/data.txt for demonstration audit tests")
class DemoFileTests(unittest.TestCase):
    def setUp(self):
        import h5py
        import numpy as np

        self.h5py = h5py
        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / (
            "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_"
            "demo.hdf5"
        )

    def add_episode(self, data, index, steps=3):
        np = self.np
        group = data.create_group(f"demo_{index}")
        group.attrs["num_samples"] = steps
        group.attrs["init_state"] = np.array([index, index + 0.5], dtype=np.float64)
        group.create_dataset("actions", data=np.zeros((steps, 7), dtype=np.float64))
        group.create_dataset("states", data=np.zeros((steps, 8), dtype=np.float64))
        group.create_dataset("robot_states", data=np.zeros((steps, 9), dtype=np.float64))
        group.create_dataset("rewards", data=np.zeros(steps, dtype=np.float64))
        done = np.zeros(steps, dtype=np.uint8)
        done[-1] = 1
        group.create_dataset("dones", data=done)
        obs = group.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((steps, 4, 4, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((steps, 4, 4, 3), dtype=np.uint8))
        obs.create_dataset("ee_states", data=np.zeros((steps, 6), dtype=np.float64))
        obs.create_dataset("gripper_states", data=np.zeros((steps, 2), dtype=np.float64))
        obs.create_dataset("joint_states", data=np.zeros((steps, 7), dtype=np.float64))
        return group

    def create_file(self, count=6):
        with self.h5py.File(self.path, "w") as h5:
            data = h5.create_group("data")
            data.attrs["num_demos"] = count
            data.attrs["total"] = count * 3
            data.attrs["bddl_file_name"] = (
                "libero_90/KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_"
                "in_the_middle.bddl"
            )
            data.attrs["problem_info"] = (
                '{"language_instruction": "stack the front black bowl on the middle bowl"}'
            )
            for index in range(count):
                self.add_episode(data, index)

    def test_complete_file_reports_schema_and_counts(self):
        self.create_file()
        task = {"bytes": self.path.stat().st_size, "sha256": file_hash(self.path),
                "expected_episodes": 6}
        result = inspect_demo_file(
            self.path, "stack", task, {"action_dim": 7, "image_shape": [4, 4, 3]}
        )
        self.assertEqual(result["summary"]["valid_episodes"], 6)
        self.assertEqual(result["summary"]["transitions"], 18)
        self.assertEqual(result["summary"]["excluded_episodes"], 0)
        self.assertEqual(result["summary"]["source_seeds_present"], 0)

    def test_bad_action_and_length_are_excluded(self):
        with self.h5py.File(self.path, "w") as h5:
            data = h5.create_group("data")
            group = self.add_episode(data, 0)
            group["actions"][1, 0] = self.np.nan
            del group["obs/joint_states"]
            group["obs"].create_dataset(
                "joint_states", data=self.np.zeros((2, 7), dtype=self.np.float64)
            )
        with self.h5py.File(self.path, "r") as h5:
            result = inspect_episode(
                h5["data/demo_0"], "demo_0", {"action_dim": 7, "image_shape": [4, 4, 3]},
                "KITCHEN_SCENE2",
            )
        self.assertEqual(result["status"], "excluded")
        self.assertIn("nonfinite:actions", result["reasons"])
        self.assertIn("length_mismatch:obs/joint_states", result["reasons"])

    def test_source_checksum_is_enforced(self):
        self.create_file()
        task = {"bytes": self.path.stat().st_size, "sha256": "0" * 64,
                "expected_episodes": 6}
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            inspect_demo_file(
                self.path, "stack", task, {"action_dim": 7, "image_shape": [4, 4, 3]}
            )


if __name__ == "__main__":
    unittest.main()
