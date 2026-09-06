"""Deterministic LIBERO smoke rollout and artifact verification utilities.

Heavy simulation imports are intentionally local to runtime functions so the
core package and its unit tests stay usable without the Linux-only stack.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import random
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


class LiberoSmokeConfigError(ValueError):
    """Raised when the Day 3 rollout configuration is invalid."""


@dataclass(frozen=True, slots=True)
class LiberoSmokeConfig:
    suite: str
    task_id: int
    init_state_index: int
    seed: int
    max_steps: int
    control_frequency: int
    settle_steps: int
    camera_names: tuple[str, ...]
    width: int
    height: int
    video_fps: int
    output_dir: Path


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise LiberoSmokeConfigError(f"[{key}] must be a TOML table")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise LiberoSmokeConfigError(f"unknown key(s) in {location}: {', '.join(unknown)}")


def _integer(
    table: dict[str, Any], key: str, location: str, *, minimum: int, maximum: int
) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise LiberoSmokeConfigError(
            f"{location}.{key} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise LiberoSmokeConfigError(f"{field} must be a non-empty string")
    normalized = value.replace("\\", "/")
    if PurePosixPath(normalized).is_absolute() or PureWindowsPath(value).is_absolute():
        raise LiberoSmokeConfigError(f"{field} must be relative to the repository root")
    if ".." in PurePosixPath(normalized).parts:
        raise LiberoSmokeConfigError(f"{field} must not escape the repository root")
    return Path(value)


def load_libero_smoke_config(path: str | Path) -> LiberoSmokeConfig:
    """Load and strictly validate a Day 3 LIBERO rollout config."""

    config_path = Path(path)
    if not config_path.is_file():
        raise LiberoSmokeConfigError(f"config file does not exist: {config_path}")
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise LiberoSmokeConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    _reject_unknown(raw, {"task", "episode", "render", "artifacts"}, "root")
    task = _table(raw, "task")
    episode = _table(raw, "episode")
    render = _table(raw, "render")
    artifacts = _table(raw, "artifacts")
    _reject_unknown(task, {"suite", "task_id", "init_state_index"}, "[task]")
    _reject_unknown(
        episode, {"seed", "max_steps", "control_frequency", "settle_steps"}, "[episode]"
    )
    _reject_unknown(render, {"camera_names", "width", "height", "video_fps"}, "[render]")
    _reject_unknown(artifacts, {"output_dir"}, "[artifacts]")

    suite = task.get("suite")
    if not isinstance(suite, str) or not suite.startswith("libero_"):
        raise LiberoSmokeConfigError("task.suite must be a LIBERO suite name")

    camera_names = render.get("camera_names")
    if (
        not isinstance(camera_names, list)
        or not camera_names
        or not all(isinstance(name, str) and name for name in camera_names)
        or len(camera_names) != len(set(camera_names))
    ):
        raise LiberoSmokeConfigError("render.camera_names must be a non-empty unique string array")

    return LiberoSmokeConfig(
        suite=suite,
        task_id=_integer(task, "task_id", "task", minimum=0, maximum=999),
        init_state_index=_integer(
            task, "init_state_index", "task", minimum=0, maximum=1_000_000
        ),
        seed=_integer(episode, "seed", "episode", minimum=0, maximum=2**32 - 1),
        max_steps=_integer(episode, "max_steps", "episode", minimum=1, maximum=10_000),
        control_frequency=_integer(
            episode, "control_frequency", "episode", minimum=1, maximum=1_000
        ),
        settle_steps=_integer(episode, "settle_steps", "episode", minimum=0, maximum=1_000),
        camera_names=tuple(camera_names),
        width=_integer(render, "width", "render", minimum=16, maximum=4_096),
        height=_integer(render, "height", "render", minimum=16, maximum=4_096),
        video_fps=_integer(render, "video_fps", "render", minimum=1, maximum=240),
        output_dir=_safe_relative_path(artifacts.get("output_dir"), "artifacts.output_dir"),
    )


def resolve_artifact_root(config: LiberoSmokeConfig, repo_root: str | Path) -> Path:
    """Resolve the artifact root and prove it remains inside the repository."""

    root = Path(repo_root).resolve()
    output = (root / config.output_dir).resolve()
    if output != root and root not in output.parents:
        raise LiberoSmokeConfigError("resolved artifact directory escapes the repository root")
    return output


def sha256_file(path: str | Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe_value(value: Any) -> Any:
    """Describe a nested observation without copying its array payload."""

    if isinstance(value, Mapping):
        return {str(key): describe_value(item) for key, item in sorted(value.items())}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {"shape": list(shape), "dtype": str(dtype)}
    return {"type": type(value).__name__}


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars and paths without silently stringifying objects."""

    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _package_versions(names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def prepare_libero_config(artifact_root: Path) -> Path:
    """Create LIBERO's config non-interactively using packaged benchmark data."""

    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hf-libero is not installed in the active environment")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    required = (benchmark_root / "bddl_files", benchmark_root / "init_files")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(f"LIBERO benchmark data is missing: {', '.join(missing)}")

    config_dir = artifact_root / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    values = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": package_root / "datasets",
        "assets": benchmark_root / "assets",
    }
    lines = [f"{key}: {json.dumps(str(value))}" for key, value in values.items()]
    config_file = config_dir / "config.yaml"
    config_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_file


def _flatten_robot_state(observation: Mapping[str, Any], np: Any) -> Any:
    state = observation["robot_state"]
    arrays = (
        state["eef"]["pos"],
        state["eef"]["quat"],
        state["eef"]["mat"],
        state["gripper"]["qpos"],
        state["gripper"]["qvel"],
        state["joints"]["pos"],
        state["joints"]["vel"],
    )
    return np.concatenate([np.asarray(array).reshape(-1) for array in arrays]).astype(np.float64)


def _unique_run_dir(artifact_root: Path, config: LiberoSmokeConfig) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}_{config.suite}_task{config.task_id:02d}_seed{config.seed}"
    candidate = artifact_root / stem
    suffix = 1
    while candidate.exists():
        candidate = artifact_root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def run_libero_smoke(
    config: LiberoSmokeConfig, repo_root: str | Path, *, env_factory: Any = None,
    task_metadata: dict[str, Any] | None = None,
) -> Path:
    """Run one deterministic no-op episode and save video, trace, and metadata."""

    if platform.system() != "Linux":
        raise RuntimeError("LIBERO smoke rollout must run on Linux or WSL2")

    artifact_root = resolve_artifact_root(config, repo_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    libero_config_file = prepare_libero_config(artifact_root)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    suites = benchmark.get_benchmark_dict()
    if config.suite not in suites:
        raise LiberoSmokeConfigError(
            f"unknown suite {config.suite!r}; available: {', '.join(sorted(suites))}"
        )
    task_suite = suites[config.suite]()
    if not 0 <= config.task_id < len(task_suite.tasks):
        raise LiberoSmokeConfigError(
            f"task_id {config.task_id} is outside {config.suite}'s 0..{len(task_suite.tasks) - 1}"
        )

    run_dir = _unique_run_dir(artifact_root, config)
    video_path = run_dir / "episode.mp4"
    trajectory_path = run_dir / "trajectory.npz"
    first_frame_path = run_dir / "first_frame.png"
    metadata_path = run_dir / "metadata.json"
    env = env_factory(config) if env_factory is not None else LiberoEnv(
        task_suite=task_suite,
        task_id=config.task_id,
        task_suite_name=config.suite,
        episode_length=config.max_steps,
        camera_name=config.camera_names,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=config.width,
        observation_height=config.height,
        visualization_width=config.width,
        visualization_height=config.height,
        init_states=True,
        episode_index=config.init_state_index,
        num_steps_wait=config.settle_steps,
        control_freq=config.control_frequency,
        hard_reset=True,
    )

    actions: list[Any] = []
    rewards: list[float] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    success_flags: list[bool] = []
    native_flags: list[bool] = []
    stable_flags: list[bool] = []
    released_flags: list[bool] = []
    robot_states: list[Any] = []
    frame_count = 0
    final_info: dict[str, Any] = {}
    started_at = datetime.now(UTC)

    try:
        observation, reset_info = env.reset(seed=config.seed)
        observation_schema = describe_value(observation)
        robot_states.append(_flatten_robot_state(observation, np))
        initial_frame = np.ascontiguousarray(env.render(), dtype=np.uint8)
        imageio.imwrite(first_frame_path, initial_frame)

        with imageio.get_writer(
            video_path,
            fps=config.video_fps,
            codec="libx264",
            macro_block_size=None,
        ) as writer:
            writer.append_data(initial_frame)
            frame_count += 1
            for step_index in range(config.max_steps):
                action = np.asarray(get_libero_dummy_action(), dtype=np.float32)
                observation, reward, terminated, env_truncated, info = env.step(action)
                runner_truncated = step_index + 1 >= config.max_steps and not terminated
                truncated = bool(env_truncated or runner_truncated)

                actions.append(action.copy())
                rewards.append(float(reward))
                terminated_flags.append(bool(terminated))
                truncated_flags.append(truncated)
                success_flags.append(bool(info.get("is_success", False)))
                native_flags.append(bool(info.get("native_success", info.get("is_success", False))))
                stable_flags.append(bool(info.get("stable_success", False)))
                released_flags.append(bool(info.get("target_released", False)))
                robot_states.append(_flatten_robot_state(observation, np))
                writer.append_data(np.ascontiguousarray(env.render(), dtype=np.uint8))
                frame_count += 1
                final_info = dict(info)
                if (step_index + 1) % 100 == 0:
                    print(f"{env.task}: {step_index + 1}/{config.max_steps} steps", flush=True)
                if terminated or env_truncated:
                    break
    finally:
        env.close()

    np.savez_compressed(
        trajectory_path,
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_flags, dtype=np.bool_),
        truncated=np.asarray(truncated_flags, dtype=np.bool_),
        is_success=np.asarray(success_flags, dtype=np.bool_),
        native_success=np.asarray(native_flags, dtype=np.bool_),
        stable_success=np.asarray(stable_flags, dtype=np.bool_),
        target_released=np.asarray(released_flags, dtype=np.bool_),
        robot_state=np.asarray(robot_states, dtype=np.float64),
    )

    finished_at = datetime.now(UTC)
    action_array = np.asarray(actions, dtype=np.float32)
    reward_array = np.asarray(rewards, dtype=np.float32)
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "Day 3 deterministic simulator and offscreen-render smoke episode",
        "config": {
            **asdict(config),
            "camera_names": list(config.camera_names),
            "output_dir": str(config.output_dir),
        },
        "task": {
            "suite": config.suite,
            "task_id": config.task_id,
            "name": env.task,
            "language": env.task_description,
            "init_state_index": config.init_state_index,
        },
        "episode": {
            "policy": "fixed_noop",
            "step_count": len(actions),
            "frame_count": frame_count,
            "return": float(reward_array.sum()) if reward_array.size else 0.0,
            "success": bool(any(success_flags)),
            "terminated": bool(terminated_flags[-1]) if terminated_flags else False,
            "truncated": bool(truncated_flags[-1]) if truncated_flags else False,
            "reset_info": reset_info,
            "final_info": final_info,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": finished_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
        },
        "interfaces": {
            "observation": observation_schema,
            "action": {
                "shape": list(action_array.shape[1:]) if action_array.ndim == 2 else [7],
                "dtype": str(action_array.dtype),
                "low": -1.0,
                "high": 1.0,
            },
            "reward": {"shape": list(reward_array.shape), "dtype": str(reward_array.dtype)},
            "terminated": {"shape": [len(terminated_flags)], "dtype": "bool"},
            "truncated": {"shape": [len(truncated_flags)], "dtype": "bool"},
            "robot_state": {
                "shape": list(np.asarray(robot_states).shape),
                "dtype": str(np.asarray(robot_states).dtype),
            },
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "mujoco_gl": os.environ["MUJOCO_GL"],
            "pyopengl_platform": os.environ["PYOPENGL_PLATFORM"],
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": _package_versions(
                ("torch", "torchvision", "lerobot", "hf-libero", "mujoco", "robosuite", "imageio")
            ),
        },
        "artifacts": {
            "video": video_path.name,
            "trajectory": trajectory_path.name,
            "first_frame": first_frame_path.name,
            "libero_config": str(libero_config_file.relative_to(artifact_root)),
        },
    }
    metadata["artifacts"]["sha256"] = {
        "video": sha256_file(video_path),
        "trajectory": sha256_file(trajectory_path),
        "first_frame": sha256_file(first_frame_path),
    }
    if task_metadata is not None:
        metadata["purpose"] = "Day 4 four-task adapter smoke episode (no learned policy)"
        metadata["task"].update(task_metadata)
        metadata["episode"]["events"] = getattr(env, "events", [])
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def verify_episode_artifacts(
    metadata_path: str | Path, *, decode_video: bool = True
) -> dict[str, Any]:
    """Verify hashes, trajectory lengths, and optionally every video frame."""

    import imageio.v2 as imageio
    import numpy as np

    path = Path(metadata_path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    run_dir = path.parent
    artifacts = metadata["artifacts"]
    video_path = run_dir / artifacts["video"]
    trajectory_path = run_dir / artifacts["trajectory"]
    first_frame_path = run_dir / artifacts["first_frame"]
    expected_hashes = artifacts["sha256"]
    actual_hashes = {
        "video": sha256_file(video_path),
        "trajectory": sha256_file(trajectory_path),
        "first_frame": sha256_file(first_frame_path),
    }
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            f"artifact hash mismatch: expected={expected_hashes}, actual={actual_hashes}"
        )

    with np.load(trajectory_path, allow_pickle=False) as trace:
        lengths = {name: int(trace[name].shape[0]) for name in trace.files}
        expected_steps = int(metadata["episode"]["step_count"])
        for name in ("actions", "rewards", "terminated", "truncated", "is_success"):
            if lengths[name] != expected_steps:
                raise RuntimeError(
                    f"trajectory {name} has {lengths[name]} rows; expected {expected_steps}"
                )
        if lengths["robot_state"] != expected_steps + 1:
            raise RuntimeError("robot_state must contain reset state plus one state per action")

    decoded_frames = None
    frame_shape = None
    if decode_video:
        decoded_frames = 0
        with imageio.get_reader(video_path) as reader:
            for frame in reader:
                decoded_frames += 1
                frame_shape = list(frame.shape)
        if decoded_frames != int(metadata["episode"]["frame_count"]):
            raise RuntimeError(
                f"video has {decoded_frames} frames; expected {metadata['episode']['frame_count']}"
            )

    return {
        "status": "verified",
        "metadata": str(path),
        "steps": int(metadata["episode"]["step_count"]),
        "frames": decoded_frames,
        "frame_shape": frame_shape,
        "success": bool(metadata["episode"]["success"]),
        "hashes": actual_hashes,
    }


def rollout_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the deterministic Day 3 LIBERO smoke episode")
    parser.add_argument(
        "config", nargs="?", type=Path, default=Path("configs/tasks/libero_smoke.toml")
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        config = load_libero_smoke_config(args.config)
        metadata_path = run_libero_smoke(config, args.repo_root)
    except (LiberoSmokeConfigError, RuntimeError) as exc:
        parser.exit(2, f"LIBERO smoke error: {exc}\n")
    print(f"complete: {metadata_path}")
    return 0


def replay_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify and replay a saved LIBERO episode offscreen"
    )
    parser.add_argument("metadata", type=Path, help="Path to the episode metadata.json")
    parser.add_argument(
        "--skip-video-decode",
        action="store_true",
        help="Verify hashes and trace without decoding MP4",
    )
    args = parser.parse_args(argv)
    try:
        result = verify_episode_artifacts(args.metadata, decode_video=not args.skip_video_decode)
    except (OSError, KeyError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"episode verification error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
