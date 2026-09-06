"""Strict, dependency-free loading for shared project configuration."""

from __future__ import annotations

import argparse
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

KNOWN_TASKS = frozenset({"pick_place", "stack", "open_drawer", "shelf_place"})
_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DEVICE_PATTERN = re.compile(r"^(?:auto|cpu|cuda(?::\d+)?)$")


class ConfigError(ValueError):
    """Raised when a project configuration violates the frozen schema."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    seed: int
    device: str
    output_dir: Path
    deterministic: bool


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    name: str
    tasks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AppConfig:
    run: RunConfig
    project: ProjectConfig


def _table(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a TOML table")
    return value


def _reject_unknown(table: dict[str, Any], allowed: set[str], location: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"unknown key(s) in {location}: {', '.join(unknown)}")


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ConfigError(f"{field} must be relative to the repository root")
    if ".." in PurePosixPath(value.replace("\\", "/")).parts:
        raise ConfigError(f"{field} must not escape the repository root")
    return Path(value)


def load_config(path: str | Path) -> AppConfig:
    """Load and validate a project TOML file."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")

    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config root must be a TOML table")
    _reject_unknown(raw, {"run", "project"}, "root")

    run = _table(raw, "run")
    project = _table(raw, "project")
    _reject_unknown(run, {"seed", "device", "output_dir", "deterministic"}, "[run]")
    _reject_unknown(project, {"name", "tasks"}, "[project]")

    seed = run.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ConfigError("run.seed must be an integer in [0, 2^32 - 1]")

    device = run.get("device")
    if not isinstance(device, str) or not _DEVICE_PATTERN.fullmatch(device):
        raise ConfigError("run.device must be auto, cpu, cuda, or cuda:<non-negative index>")

    deterministic = run.get("deterministic")
    if not isinstance(deterministic, bool):
        raise ConfigError("run.deterministic must be a boolean")

    name = project.get("name")
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ConfigError("project.name must be a lowercase kebab-case name")

    tasks = project.get("tasks")
    if not isinstance(tasks, list) or not all(isinstance(task, str) for task in tasks):
        raise ConfigError("project.tasks must be an array of strings")
    if len(tasks) not in {3, 4}:
        raise ConfigError("project.tasks must contain 3 or 4 tasks per the scope contract")
    if len(tasks) != len(set(tasks)):
        raise ConfigError("project.tasks must not contain duplicates")
    unknown_tasks = sorted(set(tasks) - KNOWN_TASKS)
    if unknown_tasks:
        raise ConfigError(f"unknown project task(s): {', '.join(unknown_tasks)}")

    return AppConfig(
        run=RunConfig(
            seed=seed,
            device=device,
            output_dir=_safe_relative_path(run.get("output_dir"), "run.output_dir"),
            deterministic=deterministic,
        ),
        project=ProjectConfig(name=name, tasks=tuple(tasks)),
    )


def resolve_output_dir(config: AppConfig, repo_root: str | Path) -> Path:
    """Resolve the configured output directory and prove it stays in the repository."""

    root = Path(repo_root).resolve()
    output = (root / config.run.output_dir).resolve()
    if output != root and root not in output.parents:
        raise ConfigError("resolved output directory escapes the repository root")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a Robust VLA Recovery config")
    parser.add_argument("config", type=Path, help="TOML configuration to validate")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        output = resolve_output_dir(config, Path.cwd())
    except ConfigError as exc:
        parser.exit(2, f"configuration error: {exc}\n")

    print(
        f"valid: project={config.project.name} seed={config.run.seed} "
        f"device={config.run.device} tasks={len(config.project.tasks)} output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
