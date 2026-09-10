import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from robust_vla_recovery.data.conversion import (
    SPECIAL_TOKENS,
    build_vocabulary,
    encode_language,
    load_conversion_config,
    select_sample_records,
    validate_training_dataset,
    write_training_dataset,
)
from robust_vla_recovery.data.demo_audit import SPLITS, TASKS
from robust_vla_recovery.data.rollout import file_hash

DATA_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "h5py"))
REPO = Path(__file__).resolve().parents[1]


class ConversionConfigTests(unittest.TestCase):
    def test_public_config_freezes_alignment_and_storage(self):
        config = load_conversion_config(REPO / "configs/data_conversion.toml")
        self.assertEqual(config["control_frequency"], 20)
        self.assertEqual(config["state_fields"], [
            "ee_states", "gripper_states", "joint_states"
        ])
        self.assertEqual(config["compression"], {"codec": "gzip", "level": 1})

    def test_vocabulary_uses_training_language_only(self):
        rows = [
            {"task": "pick_place", "split": "train"},
            {"task": "stack", "split": "test"},
        ]
        languages = {
            "pick_place": "Pick up soup.",
            "stack": "Never-seen token",
        }
        tokens, vocabulary = build_vocabulary(languages, rows)
        self.assertEqual(tuple(tokens[:4]), SPECIAL_TOKENS)
        self.assertNotIn("never", vocabulary)
        self.assertIn(vocabulary["<unk>"], encode_language(languages["stack"], vocabulary))

    def test_sample_selection_rotates_tasks_and_splits(self):
        records = []
        for index, task in enumerate(TASKS):
            for split in SPLITS:
                records.append({
                    "index": len(records), "task": task, "split": split,
                    "episode_id": f"demo_{index}",
                })
        selected = select_sample_records(records, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual({row["task"] for row in selected}, set(TASKS))
        self.assertEqual({row["split"] for row in selected}, set(SPLITS))


@unittest.skipUnless(DATA_AVAILABLE, "install requirements/data.txt for conversion tests")
class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        import h5py
        import numpy as np

        self.h5py = h5py
        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "source"
        self.dataset_root.mkdir()
        self.rows = []
        self.results = {}
        for task_index, task in enumerate(TASKS):
            filename = f"{task}.hdf5"
            with h5py.File(self.dataset_root / filename, "w") as h5:
                data = h5.create_group("data")
                for split_index, split in enumerate(SPLITS):
                    episode_id = f"demo_{split_index}"
                    group = data.create_group(episode_id)
                    steps = split_index + 2
                    group.create_dataset(
                        "actions",
                        data=np.full((steps, 7), task_index / 10, dtype=np.float64),
                    )
                    obs = group.create_group("obs")
                    pixels = np.full(
                        (steps, 4, 4, 3), task_index * 10 + split_index, dtype=np.uint8
                    )
                    obs.create_dataset("agentview_rgb", data=pixels)
                    obs.create_dataset("eye_in_hand_rgb", data=pixels + 1)
                    obs.create_dataset("ee_states", data=np.zeros((steps, 6)))
                    obs.create_dataset("gripper_states", data=np.zeros((steps, 2)))
                    obs.create_dataset("joint_states", data=np.zeros((steps, 7)))
                    self.rows.append({
                        "task": task,
                        "episode_id": episode_id,
                        "split": split,
                        "steps": steps,
                        "source_seed": None,
                        "scene_id": f"SCENE{task_index}",
                        "scene_key": f"SCENE{task_index}:state{split_index}",
                        "split_group": f"group{task_index}-{split_index}",
                        "source_file": filename,
                    })
            path = self.dataset_root / filename
            self.results[task] = {
                "file": filename,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
                "language": f"perform task {task}",
            }
        self.audit = {
            "source": {"repository": "test", "revision": "0" * 40},
            "split": {"seed": 378, "train": 1, "validation": 1, "test": 1},
            "expected": {"action_dim": 7, "image_shape": [4, 4, 3]},
        }
        self.config = {
            "control_frequency": 20,
            "compression": {"codec": "gzip", "level": 1},
            "tokenizer": {"name": "lowercase-wordpunct-v1", "lowercase": True},
        }

    def write(self, name):
        path = self.root / name
        fields = write_training_dataset(
            path,
            self.dataset_root,
            self.audit,
            self.results,
            self.rows,
            self.config,
        )
        manifest = {
            "schema_version": 1,
            "format": {
                "name": "rvla-policy-training-v1",
                "control_frequency": 20,
                "tokenizer": self.config["tokenizer"],
            },
            **fields,
        }
        return path, manifest

    def test_two_writes_have_identical_checksums_and_boundaries(self):
        first, manifest = self.write("first.h5")
        second, second_manifest = self.write("second.h5")
        self.assertEqual(manifest, second_manifest)
        self.assertEqual(file_hash(first), file_hash(second))
        result = validate_training_dataset(first, manifest)
        self.assertEqual(result["episodes"], 12)
        with self.h5py.File(first, "r") as h5:
            lengths = h5["episode_lengths"][()]
            offsets = h5["episode_offsets"][()]
            self.assertTrue(self.np.array_equal(self.np.diff(offsets), lengths))
            self.assertEqual(int(offsets[-1]), sum(row["steps"] for row in self.rows))

    def test_language_frame_misalignment_is_rejected(self):
        path, manifest = self.write("misaligned.h5")
        with self.h5py.File(path, "r+") as h5:
            h5["episodes/000000/observation/language_index"][0] = 1
        with self.assertRaisesRegex(ValueError, "language alignment"):
            validate_training_dataset(path, manifest)

    def test_episode_group_leakage_is_rejected(self):
        path, manifest = self.write("leak.h5")
        manifest = json.loads(json.dumps(manifest))
        validation = next(
            row for row in manifest["episodes"] if row["split"] == "validation"
        )
        validation["split_group"] = manifest["episodes"][0]["split_group"]
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_training_dataset(path, manifest)


if __name__ == "__main__":
    unittest.main()
