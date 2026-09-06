import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from robust_vla_recovery.data.collection import (
    collect, collection_requests, load_collection, storage_estimate, verify_batch,
)
from robust_vla_recovery.data.rollout import EpisodeWriter, file_hash, verify_episode, write_json

DATA_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "h5py"))
REPO = Path(__file__).resolve().parents[1]


class CollectionConfigTests(unittest.TestCase):
    def test_twenty_episode_manifest_and_storage_budget(self):
        config, catalog = load_collection(REPO / "configs/rollout.toml")
        estimate = storage_estimate(config, catalog)
        self.assertEqual(estimate["episodes"], 20)
        self.assertEqual(estimate["rgb_frames"], 16040)
        self.assertEqual(estimate["raw_rgb_bytes"], 3153592320)

    def test_truncated_batch_matrix_is_rejected(self):
        config, _ = load_collection(REPO / "configs/rollout.toml")
        requests = collection_requests(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.json"
            write_json(path, {"schema_version": 1, "status": "complete", "config": config,
                              "requests": requests[:-1], "results": [{}] * 19})
            with self.assertRaisesRegex(ValueError, "matrix"):
                verify_batch(path)


@unittest.skipUnless(DATA_AVAILABLE, "install requirements/data.txt for logger tests")
class RolloutTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name) / "episode"
        self.provenance = {"episode_id": "episode", "task_key": "stack", "task_name": "example",
                           "libero_task_id": 16, "seed": 42, "init_state_index": 0,
                           "instruction_index": 0, "perturbation": {"enabled": False}, "horizon": 2}
        self.flags = {key: False for key in ("is_success", "native_success", "stable_success")}
        self.flags["target_released"] = True

    def observation(self, offset=0):
        np = self.np
        return {"pixels": {"image": np.full((8, 8, 3), offset, dtype=np.uint8),
                           "image2": np.full((8, 8, 3), 200 - offset, dtype=np.uint8)},
                "robot_state": {
                    "eef": {"pos": np.full(3, offset), "quat": np.ones(4), "mat": np.eye(3)},
                    "gripper": {"qpos": np.zeros(2), "qvel": np.ones(2)},
                    "joints": {"pos": np.arange(7), "vel": np.zeros(7)},
                }}

    def writer(self):
        return EpisodeWriter(self.directory, self.provenance, self.observation(), "원래 지시")

    def append(self, writer, index, *, observation=None, end=False):
        writer.append(index, self.np.array([0, 0, 0, 0, 0, 0, -1]), float(index),
                      observation or self.observation(index + 1), "다음 지시",
                      terminated=False, truncated=end, info=self.flags)

    def completed(self):
        with self.writer() as writer:
            self.append(writer, 0)
            self.append(writer, 1, end=True)
            return writer.finish(final_info={"is_success": False}, events=[])

    def refresh_file_hash(self, metadata):
        value = json.loads(metadata.read_text())
        value["sha256"] = file_hash(metadata.parent / "episode.h5")
        write_json(metadata, value)

    def completed_batch(self):
        config = {"tasks": ["stack"], "seeds": [42], "init_state_indices": [0],
                  "instruction_indices": [0], "perturbation": {"enabled": False}}
        requests = collection_requests(config)
        self.provenance.update(requests[0])
        self.directory = Path(self.temporary.name) / requests[0]["episode_id"]
        metadata = self.completed()
        batch_path = self.directory.parent / "batch.json"
        write_json(batch_path, {"schema_version": 1, "status": "complete", "config": config,
                               "requests": requests, "results": [verify_episode(metadata)]})
        return batch_path, metadata

    def test_batch_accepts_original_collection_checksum(self):
        batch_path, _ = self.completed_batch()
        self.assertEqual(verify_batch(batch_path)["episodes"], 1)

    def test_batch_rejects_missing_or_invalid_original_checksum(self):
        batch_path, _ = self.completed_batch()
        batch = json.loads(batch_path.read_text())
        for value in (None, "bad", "g" * 64, "0" * 64):
            with self.subTest(value=value):
                batch["results"][0]["sha256"] = value
                if value is None:
                    del batch["results"][0]["sha256"]
                write_json(batch_path, batch)
                with self.assertRaisesRegex(ValueError, "original collection checksum"):
                    verify_batch(batch_path)

    def test_batch_rejects_changed_episode_with_matching_updated_metadata(self):
        import h5py

        batch_path, metadata = self.completed_batch()
        with h5py.File(self.directory / "episode.h5", "r+") as h5:
            h5["transitions/reward"][0] = 0.5
        self.refresh_file_hash(metadata)
        self.assertEqual(verify_episode(metadata)["status"], "verified")
        with self.assertRaisesRegex(ValueError, "original collection checksum mismatch"):
            verify_batch(batch_path)

    def check_collection_cleanup_failure(self, *, step_failure):
        config_path = REPO / "configs/rollout.toml"
        config, catalog = load_collection(config_path)
        config.update(output_dir=self.temporary.name, tasks=["stack"], seeds=[42, 43],
                      init_state_indices=[0, 1], instruction_indices=[0, 0], width=8, height=8)
        catalog["stack"] = replace(catalog["stack"], horizon=1)
        test = self
        instances = []

        class Backend:
            task_id = 16
            bddl_sha256 = "0" * 64

            def __init__(self, *args, **kwargs):
                self.first = not instances
                self.closed = False
                instances.append(self)

            def reset(self, **kwargs):
                if not self.first:
                    # The first result must be persisted before the next episode starts.
                    path = next(Path(test.temporary.name).glob("batch_*/batch.json"))
                    saved = json.loads(path.read_text())
                    test.assertEqual(len(saved["results"]), 1)
                    test.assertIn("cleanup_error", saved["results"][0])
                return test.observation(), {}

            def step(self, action):
                if self.first and step_failure:
                    raise ValueError("step failed")
                return test.observation(1), 0.0, False, False, {"is_success": False}

            def target_released(self):
                return True

            def close(self):
                self.closed = True
                if self.first:
                    raise RuntimeError("close failed")

        with (patch("robust_vla_recovery.data.collection.load_collection",
                    return_value=(config, catalog)),
              patch("robust_vla_recovery.data.collection.platform.system", return_value="Linux"),
              patch("robust_vla_recovery.data.collection.prepare_libero_config"),
              patch("robust_vla_recovery.data.collection.importlib.metadata.version",
                    return_value="test"),
              patch("robust_vla_recovery.envs.libero_tasks.LiberoTaskBackend", Backend),
              patch.dict(os.environ), redirect_stdout(io.StringIO())):
            with self.assertRaisesRegex(RuntimeError, "failed episodes"):
                collect(config_path)
        batch_path = next(Path(self.temporary.name).glob("batch_*/batch.json"))
        batch = json.loads(batch_path.read_text())
        self.assertEqual(batch["status"], "failed")
        self.assertEqual(len(batch["results"]), 2)
        first, second = batch["results"]
        self.assertEqual(first["status"], "error")
        self.assertIn("close failed", first["cleanup_error"])
        if step_failure:
            self.assertIn("step failed", first["error"])
            self.assertEqual(first["events"][0]["type"], "simulator_error")
        else:
            self.assertNotIn("error", first)
            self.assertEqual(first["sha256"], verify_episode(
                batch_path.parent / first["episode_id"] / "metadata.json")["sha256"])
        self.assertEqual(second["status"], "verified")
        self.assertTrue(all(backend.closed for backend in instances))

    def test_cleanup_failure_persists_result_and_continues_collection(self):
        self.check_collection_cleanup_failure(step_failure=False)

    def test_cleanup_failure_preserves_primary_error_and_continues_collection(self):
        self.check_collection_cleanup_failure(step_failure=True)

    def test_round_trip_keeps_lossless_rgb_state_language_and_alignment(self):
        import h5py

        metadata = self.completed()
        verified = verify_episode(metadata)
        self.assertEqual(
            (verified["steps"], verified["observations"], verified["rgb_frames"]), (2, 3, 6)
        )
        with h5py.File(metadata.parent / "episode.h5") as h5:
            self.np.testing.assert_array_equal(
                h5["observations/image"][2], self.observation(2)["pixels"]["image"]
            )
            self.np.testing.assert_array_equal(
                h5["observations/timestamp_ns"][:], [0, 50_000_000, 100_000_000]
            )
            self.assertEqual(
                list(h5["observations/language"].asstr()[:]), ["원래 지시", "다음 지시", "다음 지시"]
            )
            self.assertEqual(h5["observations/proprio"][1, 0], 1)
            self.assertEqual(h5["transitions/reward"][0], 0)

    def test_skipped_and_duplicate_steps_rejected_without_partial_append(self):
        with self.writer() as writer:
            with self.assertRaisesRegex(ValueError, "skipped or duplicated"):
                self.append(writer, 1)
            self.assertEqual(writer.h5["observations/image"].shape[0], 1)
            self.append(writer, 0)
            with self.assertRaises(ValueError):
                self.append(writer, 0)
            self.append(writer, 1, end=True)
            writer.finish(final_info={}, events=[])

    def test_missing_camera_bad_shape_and_nan_are_rejected_before_write(self):
        with self.writer() as writer:
            for mode in ("missing", "shape", "nan"):
                with self.subTest(mode=mode):
                    obs = self.observation()
                    if mode == "missing":
                        del obs["pixels"]["image2"]
                    elif mode == "shape":
                        obs["pixels"]["image"] = self.np.zeros((9, 8, 3), dtype="u1")
                    else:
                        obs["robot_state"]["eef"]["pos"] = self.np.full(3, self.np.nan)
                    with self.assertRaises(ValueError):
                        self.append(writer, 0, observation=obs)
                    self.assertEqual(writer.steps, 0)

    def test_incomplete_episode_is_not_published(self):
        with self.writer() as writer:
            self.append(writer, 0)
            with self.assertRaises(RuntimeError):
                writer.finish(final_info={}, events=[])
        metadata = self.directory / "metadata.json"
        self.assertEqual(json.loads(metadata.read_text())["status"], "aborted")
        self.assertFalse((self.directory / "episode.h5").exists())
        with self.assertRaises(ValueError):
            verify_episode(metadata)

    def test_write_exception_preserves_aborted_marker(self):
        with self.writer() as writer:
            with patch.object(writer, "_row", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    self.append(writer, 0)
        value = json.loads((self.directory / "metadata.json").read_text())
        self.assertEqual(value["status"], "aborted")
        self.assertIn("disk full", value["error"])

    def test_append_after_terminal_and_directory_overwrite_are_rejected(self):
        with self.writer() as writer:
            self.append(writer, 0, end=True)
            with self.assertRaises(RuntimeError):
                self.append(writer, 1)
            writer.finish(final_info={}, events=[])
        with self.assertRaises(FileExistsError):
            self.writer()

    def test_binary_corruption_is_rejected(self):
        metadata = self.completed()
        path = self.directory / "episode.h5"
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_episode(metadata)

    def test_missing_rgb_frame_detected_even_with_updated_container_hash(self):
        import h5py

        metadata = self.completed()
        with h5py.File(self.directory / "episode.h5", "r+") as h5:
            h5["observations/image2"].resize(2, axis=0)
        self.refresh_file_hash(metadata)
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            verify_episode(metadata)

    def test_timestamp_misalignment_detected(self):
        import h5py

        metadata = self.completed()
        with h5py.File(self.directory / "episode.h5", "r+") as h5:
            h5["observations/timestamp_ns"][1] = 0
        self.refresh_file_hash(metadata)
        with self.assertRaisesRegex(ValueError, "alignment"):
            verify_episode(metadata)

    def test_pixel_corruption_detected_against_capture_hash(self):
        import h5py

        metadata = self.completed()
        with h5py.File(self.directory / "episode.h5", "r+") as h5:
            h5["observations/image"][1, 0, 0, 0] = 250
        self.refresh_file_hash(metadata)
        with self.assertRaisesRegex(ValueError, "capture/decode"):
            verify_episode(metadata)

    def test_metadata_provenance_mismatch_rejected(self):
        metadata = self.completed()
        value = json.loads(metadata.read_text())
        value["provenance"]["seed"] += 1
        write_json(metadata, value)
        with self.assertRaisesRegex(ValueError, "provenance"):
            verify_episode(metadata)

    def test_changed_final_success_is_rejected(self):
        metadata = self.completed()
        value = json.loads(metadata.read_text())
        value["final_info"]["is_success"] = True
        write_json(metadata, value)
        with self.assertRaisesRegex(ValueError, "final outcome"):
            verify_episode(metadata)

    def test_native_success_round_trip_with_numpy_boolean_metadata(self):
        flags = {"is_success": self.np.bool_(True), "native_success": self.np.bool_(True),
                 "stable_success": self.np.bool_(True), "target_released": self.np.bool_(True)}
        with self.writer() as writer:
            writer.append(0, self.np.zeros(7), 1.0, self.observation(1), "성공 지시",
                          terminated=True, truncated=False, info=flags)
            metadata = writer.finish(final_info=flags, events=[])
        self.assertTrue(verify_episode(metadata)["success"])


if __name__ == "__main__":
    unittest.main()
