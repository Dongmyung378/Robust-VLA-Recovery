"""Streaming lossless episode storage with an explicit T / T+1 contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

CAMERAS = ("image", "image2")
STATE_FIELDS = (
    ("eef", "pos", (3,)), ("eef", "quat", (4,)), ("eef", "mat", (3, 3)),
    ("gripper", "qpos", (2,)), ("gripper", "qvel", (2,)),
    ("joints", "pos", (7,)), ("joints", "vel", (7,)),
)
FLAGS = ("terminated", "truncated", "is_success", "native_success",
         "stable_success", "target_released")


def scalar_json(value: Any) -> Any:
    if callable(getattr(value, "item", None)):
        return value.item()
    raise TypeError(f"not a JSON scalar: {type(value).__name__}")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomic completion marker; do not leave a partially written JSON manifest."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=scalar_json) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def observation_arrays(observation: dict, language: str) -> tuple[dict, Any]:
    import numpy as np

    if not isinstance(language, str) or not language.strip() or "\x00" in language:
        raise ValueError("language must be non-empty UTF-8 text without NUL")
    if set(observation["pixels"]) != set(CAMERAS):
        raise ValueError("both image and image2 must be present")
    pixels = {}
    for camera in CAMERAS:
        array = np.asarray(observation["pixels"][camera])
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(f"{camera} must be uint8 HWC RGB")
        if min(array.shape) < 1:
            raise ValueError("empty RGB frame")
        pixels[camera] = array
    fields = []
    for group, name, shape in STATE_FIELDS:
        value = np.asarray(observation["robot_state"][group][name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid proprioception: {group}.{name}")
        fields.append(value.reshape(-1))
    return pixels, np.concatenate(fields)


class EpisodeWriter:
    """Persist each transition immediately; memory does not grow with horizon.

    action[t] uses observations[t] and language[t] to produce observations[t+1].
    Rewards and success flags describe that transition's resulting state.
    """

    def __init__(self, directory: Path, provenance: dict, observation: dict, language: str,
                 *, control_frequency: int = 20, gzip_level: int = 1):
        import h5py
        import numpy as np

        if type(control_frequency) is not int or control_frequency <= 0:
            raise ValueError("control frequency must be a positive integer")
        if 1_000_000_000 % control_frequency:
            raise ValueError("control frequency must divide one billion for exact nanoseconds")
        if type(gzip_level) is not int or not 0 <= gzip_level <= 9:
            raise ValueError("invalid gzip level")
        pixels, state = observation_arrays(observation, language)
        required = {"episode_id", "task_key", "task_name", "libero_task_id", "seed",
                    "init_state_index", "instruction_index", "perturbation", "horizon"}
        if not required <= set(provenance):
            raise ValueError(f"missing provenance: {sorted(required - set(provenance))}")
        # Freeze the input to prevent a caller mutating provenance after initialization.
        encoded = json.dumps(provenance, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if type(provenance["horizon"]) is not int or provenance["horizon"] < 1:
            raise ValueError("invalid horizon")
        self.manifest = {"schema_version": 1, "status": "incomplete",
                         "provenance": json.loads(encoded), "control_frequency": control_frequency,
                         "compression": {"codec": "gzip", "level": gzip_level, "lossless": True},
                         "proprio_fields": [[a, b, list(s)] for a, b, s in STATE_FIELDS]}
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.partial = self.directory / "episode.partial.h5"
        self.path = self.directory / "episode.h5"
        self.steps = 0
        self.closed = False
        self.ended = False
        self.delta_ns = 1_000_000_000 // control_frequency
        self.rgb_hashes = {camera: hashlib.sha256() for camera in CAMERAS}
        self.h5 = h5py.File(self.partial, "x")
        self.h5.attrs["schema_version"] = 1
        self.h5.attrs["status"] = "incomplete"
        self.h5.attrs["provenance_json"] = encoded
        for camera, array in pixels.items():
            self.h5.create_dataset(f"observations/{camera}", (0, *array.shape),
                                   maxshape=(None, *array.shape), chunks=(1, *array.shape),
                                   dtype="u1", compression="gzip", compression_opts=gzip_level,
                                   shuffle=True, fletcher32=True)
        for name, shape, dtype in (
            ("observations/proprio", (34,), "f8"),
            ("observations/frame_index", (), "i8"),
            ("observations/timestamp_ns", (), "i8"),
            ("transitions/action", (7,), "f4"),
            ("transitions/reward", (), "f8"),
            ("transitions/step_index", (), "i8"),
        ):
            self.h5.create_dataset(name, (0, *shape), maxshape=(None, *shape),
                                   chunks=True, dtype=dtype, compression="gzip",
                                   compression_opts=gzip_level, fletcher32=True)
        for flag in FLAGS:
            self.h5.create_dataset(f"transitions/{flag}", (0,), maxshape=(None,),
                                   chunks=True, dtype=np.bool_)
        self.h5.create_dataset("observations/language", (0,), maxshape=(None,),
                               dtype=h5py.string_dtype("utf-8"), chunks=True)
        self._observation(0, pixels, state, language)
        self.h5.flush()
        write_json(self.directory / "metadata.json", self.manifest)

    def _row(self, name: str, value: Any) -> None:
        dataset = self.h5[name]
        index = dataset.shape[0]
        dataset.resize(index + 1, axis=0)
        dataset[index] = value

    def _observation(self, index: int, pixels: dict, state: Any, language: str) -> None:
        for camera, frame in pixels.items():
            self._row(f"observations/{camera}", frame)
            self.rgb_hashes[camera].update(frame.tobytes(order="C"))
        self._row("observations/proprio", state)
        self._row("observations/language", language)
        self._row("observations/frame_index", index)
        self._row("observations/timestamp_ns", index * self.delta_ns)

    def append(self, step_index: int, action: Any, reward: float, observation: dict,
               language: str, *, terminated: bool, truncated: bool, info: dict) -> None:
        import numpy as np

        if self.closed or self.ended:
            raise RuntimeError("cannot append after episode end/close")
        if type(step_index) is not int or step_index != self.steps:
            raise ValueError("skipped or duplicated step index")
        if self.steps >= self.manifest["provenance"]["horizon"]:
            raise ValueError("episode exceeds configured horizon")
        pixels, state = observation_arrays(observation, language)
        for camera, frame in pixels.items():
            if frame.shape != self.h5[f"observations/{camera}"].shape[1:]:
                raise ValueError(f"changed camera shape: {camera}")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all() or np.abs(action).max() > 1:
            raise ValueError("action must be a finite 7-vector in [-1, 1]")
        if not math.isfinite(float(reward)):
            raise ValueError("reward must be finite")
        values = {"terminated": terminated, "truncated": truncated,
                  **{key: info[key] for key in FLAGS[2:]}}
        if not all(isinstance(value, (bool, np.bool_)) for value in values.values()):
            raise ValueError("termination/success flags must be booleans")
        if terminated and truncated:
            raise ValueError("terminated and truncated cannot both be true in this adapter")
        try:
            self._row("transitions/action", action)
            self._row("transitions/reward", reward)
            self._row("transitions/step_index", step_index)
            for flag, value in values.items():
                self._row(f"transitions/{flag}", value)
            self._observation(step_index + 1, pixels, state, language)
            self.h5.flush()
        except Exception as exc:
            self.abort(f"write failure: {exc}")
            raise
        self.steps += 1
        self.ended = bool(terminated or truncated)

    def finish(self, *, final_info: dict, events: list) -> Path:
        if self.closed or not self.ended or self.steps == 0:
            raise RuntimeError("completion requires at least one transition and an end signal")
        validate_structure(self.h5, self.manifest, self.steps)
        self.manifest.update(
            status="complete", steps=self.steps, observations=self.steps + 1,
            final_info=final_info, events=events,
            rgb_sha256={key: value.hexdigest() for key, value in self.rgb_hashes.items()},
        )
        # Serialize before publishing so invalid metadata cannot certify an episode.
        json.dumps(self.manifest, allow_nan=False, default=scalar_json)
        self.h5.attrs["status"] = "complete"
        self.h5.flush()
        self.h5.close()
        self.closed = True
        self.partial.replace(self.path)
        self.manifest.update(hdf5="episode.h5", sha256=file_hash(self.path),
                             bytes=self.path.stat().st_size)
        write_json(self.directory / "metadata.json", self.manifest)
        return self.directory / "metadata.json"

    def abort(self, reason: str) -> None:
        if self.closed:
            return
        self.h5.close()
        self.closed = True
        self.manifest.update(status="aborted", committed_steps=self.steps, error=reason)
        write_json(self.directory / "metadata.json", self.manifest)

    def __enter__(self) -> EpisodeWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.closed:
            self.abort(str(exc) if exc else "writer exited without finish")


def validate_structure(h5: Any, metadata: dict, steps: int) -> None:
    import numpy as np

    if steps < 1 or steps > metadata["provenance"]["horizon"]:
        raise ValueError("invalid episode length")
    observations = steps + 1
    expected_obs = {*CAMERAS, "proprio", "language", "frame_index", "timestamp_ns"}
    expected_transitions = {"action", "reward", "step_index", *FLAGS}
    if set(h5["observations"]) != expected_obs or set(h5["transitions"]) != expected_transitions:
        raise ValueError("missing/unexpected dataset")
    for group, count in (("observations", observations), ("transitions", steps)):
        for dataset in h5[group].values():
            if dataset.shape[0] != count:
                raise ValueError(f"length mismatch: {dataset.name}")
    for camera in CAMERAS:
        dataset = h5[f"observations/{camera}"]
        if dataset.ndim != 4 or dataset.shape[-1] != 3 or dataset.dtype != np.uint8:
            raise ValueError("invalid RGB schema")
        if (dataset.compression != "gzip"
                or dataset.compression_opts != metadata["compression"]["level"]
                or not dataset.fletcher32 or dataset.chunks[0] != 1):
            raise ValueError("RGB compression/chunk integrity contract mismatch")
    for name, shape, dtype in (("observations/proprio", (observations, 34), "float64"),
                               ("transitions/action", (steps, 7), "float32"),
                               ("transitions/reward", (steps,), "float64")):
        if h5[name].shape != shape or h5[name].dtype != np.dtype(dtype):
            raise ValueError(f"invalid shape/dtype: {name}")
        if not np.isfinite(h5[name][:]).all():
            raise ValueError(f"nonfinite data: {name}")
    if np.abs(h5["transitions/action"][:]).max() > 1:
        raise ValueError("action out of range")
    for name, expected in (
        ("observations/frame_index", np.arange(observations, dtype=np.int64)),
        ("observations/timestamp_ns", np.arange(observations, dtype=np.int64)
         * (1_000_000_000 // metadata["control_frequency"])),
        ("transitions/step_index", np.arange(steps, dtype=np.int64)),
    ):
        if h5[name].dtype != np.int64 or not np.array_equal(h5[name][:], expected):
            raise ValueError(f"timestamp/index alignment error: {name}")
    languages = h5["observations/language"].asstr()[:]
    if any(not text.strip() for text in languages):
        raise ValueError("missing language")
    for flag in FLAGS:
        dataset = h5[f"transitions/{flag}"]
        if dataset.dtype != np.bool_ or dataset.shape != (steps,):
            raise ValueError(f"invalid flag schema: {flag}")
    terminal = h5["transitions/terminated"][:]
    truncated = h5["transitions/truncated"][:]
    ended = terminal | truncated
    if ended[:-1].any() or not ended[-1] or (terminal & truncated).any():
        raise ValueError("invalid episode boundary")
    success = h5["transitions/is_success"][:]
    native = h5["transitions/native_success"][:]
    if not np.array_equal(success, np.logical_or.accumulate(native)):
        raise ValueError("native/latched success mismatch")
    stable = h5["transitions/stable_success"][:]
    released = h5["transitions/target_released"][:]
    if (stable & ~(native & released)).any():
        raise ValueError("stable success without native success and release")


def verify_episode(metadata_path: str | Path) -> dict:
    """Read and decompress every RGB frame; compare against capture-time hashes."""
    import h5py

    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("status") != "complete":
        raise ValueError("episode is not a completed schema-v1 rollout")
    if metadata.get("hdf5") != "episode.h5":
        raise ValueError("unexpected episode filename")
    data_path = path.parent / "episode.h5"
    if file_hash(data_path) != metadata["sha256"]:
        raise ValueError("HDF5 checksum mismatch")
    with h5py.File(data_path, "r") as h5:
        if h5.attrs["status"] != "complete" or h5.attrs["schema_version"] != 1:
            raise ValueError("incomplete HDF5")
        if json.loads(h5.attrs["provenance_json"]) != metadata["provenance"]:
            raise ValueError("provenance mismatch")
        steps = metadata["steps"]
        if metadata["observations"] != steps + 1:
            raise ValueError("metadata observation count mismatch")
        validate_structure(h5, metadata, steps)
        for key in ("is_success", "native_success", "stable_success", "target_released"):
            if key in metadata["final_info"]:
                if metadata["final_info"][key] != bool(h5[f"transitions/{key}"][-1]):
                    raise ValueError(f"final outcome mismatch: {key}")
        raw_rgb_bytes = 0
        for camera in CAMERAS:
            digest = hashlib.sha256()
            for frame in h5[f"observations/{camera}"]:
                digest.update(frame.tobytes(order="C"))
                raw_rgb_bytes += frame.nbytes
            if digest.hexdigest() != metadata["rgb_sha256"][camera]:
                raise ValueError(f"RGB capture/decode checksum mismatch: {camera}")
        success = bool(h5["transitions/is_success"][-1])
    return {"status": "verified", "episode_id": metadata["provenance"]["episode_id"],
            "task_key": metadata["provenance"]["task_key"], "steps": steps,
            "observations": steps + 1, "rgb_frames": 2 * (steps + 1),
            "raw_rgb_bytes": raw_rgb_bytes, "stored_bytes": data_path.stat().st_size,
            "success": success, "sha256": metadata["sha256"]}


def export_video(metadata_path: str | Path, destination: Path) -> None:
    """Render both saved cameras side by side; H.264 is only a viewing derivative."""
    import h5py
    import imageio.v2 as imageio
    import numpy as np

    verify_episode(metadata_path)
    if destination.exists():
        raise FileExistsError(destination)
    path = Path(metadata_path)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    with h5py.File(path.parent / "episode.h5", "r") as h5:
        with imageio.get_writer(destination, fps=metadata["control_frequency"],
                                codec="libx264", macro_block_size=None) as writer:
            for left, right in zip(h5["observations/image"], h5["observations/image2"],
                                   strict=True):
                writer.append_data(np.concatenate((left[::-1, ::-1], right[::-1, ::-1]), axis=1))
