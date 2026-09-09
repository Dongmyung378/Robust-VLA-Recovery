"""Audit official LIBERO demonstrations and create leakage-safe data splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import tomllib
import uuid
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from robust_vla_recovery.data.rollout import file_hash, write_json
from robust_vla_recovery.envs.task_adapter import load_task_catalog

SPLITS = ("train", "validation", "test")
TASKS = ("pick_place", "stack", "open_drawer", "shelf_place")
SCENE_PATTERN = re.compile(r"((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)")
EPISODE_PATTERN = re.compile(r"demo_(\d+)")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_DATASETS = (
    "actions",
    "states",
    "robot_states",
    "rewards",
    "dones",
    "obs/agentview_rgb",
    "obs/eye_in_hand_rgb",
    "obs/ee_states",
    "obs/gripper_states",
    "obs/joint_states",
)


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"{field} must be repository-relative")
    if ".." in value.replace("\\", "/").split("/"):
        raise ValueError(f"{field} must not escape the repository")
    return path


def _require_exact_keys(value: dict, expected: set[str], field: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"invalid {field} keys: missing={missing}, extra={extra}")


def load_audit_config(path: str | Path) -> tuple[dict, dict]:
    """Load the frozen Day 8 source, integrity, schema, and split contract."""

    config_path = Path(path)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    _require_exact_keys(
        config,
        {
            "schema_version",
            "catalog",
            "dataset_root",
            "output_dir",
            "source",
            "expected",
            "split",
            "tasks",
        },
        "config",
    )
    if config["schema_version"] != 1:
        raise ValueError("unsupported audit schema")

    for field in ("catalog", "dataset_root", "output_dir"):
        config[field] = str(_safe_relative_path(config[field], field)).replace("\\", "/")

    source = config["source"]
    if not isinstance(source, dict):
        raise ValueError("source must be a table")
    _require_exact_keys(source, {"repository", "revision"}, "source")
    if source["repository"] != "yifengzhu-hf/LIBERO-datasets":
        raise ValueError("unexpected demonstration repository")
    if not isinstance(source["revision"], str) or not re.fullmatch(
        r"[0-9a-f]{40}", source["revision"]
    ):
        raise ValueError("source revision must be a full commit hash")

    expected = config["expected"]
    if not isinstance(expected, dict):
        raise ValueError("expected must be a table")
    _require_exact_keys(expected, {"action_dim", "image_shape"}, "expected")
    if type(expected["action_dim"]) is not int or expected["action_dim"] < 1:
        raise ValueError("expected.action_dim must be positive")
    if (
        not isinstance(expected["image_shape"], list)
        or len(expected["image_shape"]) != 3
        or not all(type(value) is int and value > 0 for value in expected["image_shape"])
    ):
        raise ValueError("expected.image_shape must contain three positive integers")

    split = config["split"]
    if not isinstance(split, dict):
        raise ValueError("split must be a table")
    _require_exact_keys(split, {"seed", *SPLITS}, "split")
    if type(split["seed"]) is not int or not 0 <= split["seed"] < 2**32:
        raise ValueError("split.seed must be a uint32")
    if not all(type(split[name]) is int and split[name] > 0 for name in SPLITS):
        raise ValueError("split counts must be positive integers")

    tasks = config["tasks"]
    if not isinstance(tasks, dict):
        raise ValueError("tasks must be a table")
    _require_exact_keys(tasks, set(TASKS), "tasks")
    catalog = load_task_catalog(config["catalog"])
    expected_episodes = sum(split[name] for name in SPLITS)
    for key in TASKS:
        item = tasks[key]
        if not isinstance(item, dict):
            raise ValueError(f"tasks.{key} must be a table")
        _require_exact_keys(item, {"file", "bytes", "sha256"}, f"tasks.{key}")
        file_path = _safe_relative_path(item["file"], f"tasks.{key}.file")
        if len(file_path.parts) != 1 or file_path.suffix != ".hdf5":
            raise ValueError(f"tasks.{key}.file must be an HDF5 filename")
        if file_path.stem != f"{catalog[key].name}_demo":
            raise ValueError(f"tasks.{key}.file does not match the task catalog")
        item["file"] = file_path.name
        if type(item["bytes"]) is not int or item["bytes"] < 1:
            raise ValueError(f"tasks.{key}.bytes must be positive")
        if not isinstance(item["sha256"], str) or not SHA256_PATTERN.fullmatch(item["sha256"]):
            raise ValueError(f"tasks.{key}.sha256 is invalid")
        item["expected_episodes"] = expected_episodes
    return config, catalog


def _source_seed(group: Any) -> int | None:
    for name in ("seed", "env_seed", "random_seed"):
        if name in group.attrs:
            value = group.attrs[name]
            if hasattr(value, "item"):
                value = value.item()
            if type(value) is int and 0 <= value < 2**32:
                return value
    return None


def _hash_array(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _finite(dataset: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(dataset[()]).all())


def inspect_episode(group: Any, episode_id: str, expected: dict, scene_id: str) -> dict:
    """Check one official demonstration without relying on simulator imports."""

    import h5py
    import numpy as np

    reasons: list[str] = []
    warnings: list[str] = []
    datasets: dict[str, Any] = {}
    for name in REQUIRED_DATASETS:
        try:
            value = group[name]
        except KeyError:
            reasons.append(f"missing_dataset:{name}")
            continue
        if not isinstance(value, h5py.Dataset):
            reasons.append(f"not_a_dataset:{name}")
            continue
        datasets[name] = value

    actions = datasets.get("actions")
    steps = int(actions.shape[0]) if actions is not None and actions.ndim >= 1 else 0
    if steps < 1:
        reasons.append("empty_actions")
    elif actions.ndim != 2 or actions.shape[1] != expected["action_dim"]:
        reasons.append("invalid_action_shape")

    if steps:
        for name, dataset in datasets.items():
            if dataset.ndim < 1 or dataset.shape[0] != steps:
                reasons.append(f"length_mismatch:{name}")

    for name in ("obs/agentview_rgb", "obs/eye_in_hand_rgb"):
        dataset = datasets.get(name)
        if dataset is not None and (
            dataset.dtype != np.uint8
            or dataset.ndim != 4
            or tuple(dataset.shape[1:]) != tuple(expected["image_shape"])
        ):
            reasons.append(f"invalid_image:{name}")

    for name in (
        "actions",
        "states",
        "robot_states",
        "rewards",
        "obs/ee_states",
        "obs/gripper_states",
        "obs/joint_states",
    ):
        dataset = datasets.get(name)
        if dataset is not None:
            try:
                if not _finite(dataset):
                    reasons.append(f"nonfinite:{name}")
            except (OSError, TypeError, ValueError):
                reasons.append(f"unreadable:{name}")

    action_min = action_max = None
    action_sha256 = None
    if actions is not None and actions.ndim == 2 and actions.shape[0]:
        try:
            action_values = actions[()]
            action_min = float(action_values.min())
            action_max = float(action_values.max())
            action_sha256 = _hash_array(action_values)
            if action_min < -1.000001 or action_max > 1.000001:
                reasons.append("action_out_of_range")
        except (OSError, TypeError, ValueError):
            reasons.append("unreadable:actions")

    dones = datasets.get("dones")
    if dones is not None and dones.ndim == 1 and dones.shape[0]:
        try:
            done_values = dones[()]
            if not np.isin(done_values, (0, 1, False, True)).all():
                reasons.append("invalid_done_values")
            elif not bool(done_values[-1]):
                warnings.append("final_done_is_false")
        except (OSError, TypeError, ValueError):
            reasons.append("unreadable:dones")

    declared_steps = group.attrs.get("num_samples")
    if hasattr(declared_steps, "item"):
        declared_steps = declared_steps.item()
    if type(declared_steps) is not int or declared_steps != steps:
        reasons.append("num_samples_mismatch")

    init_state = group.attrs.get("init_state")
    init_state_sha256 = None
    if init_state is None:
        reasons.append("missing_init_state")
    else:
        init_state = np.asarray(init_state)
        if init_state.size == 0 or not np.isfinite(init_state).all():
            reasons.append("invalid_init_state")
        else:
            init_state_sha256 = _hash_array(init_state)

    source_seed = _source_seed(group)
    seed_key = f"seed:{source_seed}" if source_seed is not None else f"state:{init_state_sha256}"
    scene_key = f"{scene_id}:{init_state_sha256}"
    return {
        "episode_id": episode_id,
        "status": "excluded" if reasons else "valid",
        "steps": steps,
        "action_shape": list(actions.shape[1:]) if actions is not None else None,
        "image_shape": list(expected["image_shape"]),
        "action_min": action_min,
        "action_max": action_max,
        "action_sha256": action_sha256,
        "source_seed": source_seed,
        "init_state_sha256": init_state_sha256,
        "scene_id": scene_id,
        "scene_key": scene_key,
        "split_group": f"{scene_key}|{seed_key}",
        "reasons": sorted(set(reasons)),
        "warnings": sorted(set(warnings)),
    }


def _length_outliers(episodes: list[dict]) -> None:
    lengths = [episode["steps"] for episode in episodes if episode["status"] == "valid"]
    if len(lengths) < 4:
        return
    center = statistics.median(lengths)
    deviation = statistics.median(abs(value - center) for value in lengths)
    threshold = max(3 * deviation, center * 0.5, 1)
    for episode in episodes:
        if episode["status"] == "valid" and abs(episode["steps"] - center) > threshold:
            episode["warnings"].append("length_outlier")
            episode["warnings"].sort()


def inspect_demo_file(path: Path, task_key: str, task: dict, expected: dict) -> dict:
    """Verify one pinned source file and summarize all demonstrations inside it."""

    import h5py

    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != task["bytes"]:
        raise ValueError(f"source size mismatch: {task_key}")
    digest = file_hash(path)
    if digest != task["sha256"]:
        raise ValueError(f"source SHA-256 mismatch: {task_key}")
    scene_match = SCENE_PATTERN.search(path.name)
    if scene_match is None:
        raise ValueError(f"scene identifier missing from filename: {task_key}")
    scene_id = scene_match.group(1)

    with h5py.File(path, "r") as h5:
        if "data" not in h5 or not isinstance(h5["data"], h5py.Group):
            raise ValueError(f"missing data group: {task_key}")
        data = h5["data"]
        episode_names = sorted(
            (name for name in data if EPISODE_PATTERN.fullmatch(name)),
            key=lambda name: int(EPISODE_PATTERN.fullmatch(name).group(1)),
        )
        unexpected = sorted(set(data) - set(episode_names))
        if unexpected:
            raise ValueError(f"unexpected data members in {task_key}: {unexpected}")
        declared_count = data.attrs.get("num_demos")
        if hasattr(declared_count, "item"):
            declared_count = declared_count.item()
        if declared_count != len(episode_names):
            raise ValueError(f"num_demos mismatch: {task_key}")
        if len(episode_names) != task["expected_episodes"]:
            raise ValueError(f"unexpected episode count: {task_key}")

        bddl_name = str(data.attrs.get("bddl_file_name", ""))
        expected_bddl = path.name.removesuffix("_demo.hdf5") + ".bddl"
        if Path(bddl_name).name != expected_bddl:
            raise ValueError(f"BDDL identity mismatch: {task_key}")
        try:
            problem = json.loads(str(data.attrs["problem_info"]))
            language = problem["language_instruction"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid problem_info: {task_key}") from exc
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"missing language instruction: {task_key}")

        episodes = [inspect_episode(data[name], name, expected, scene_id) for name in episode_names]
        _length_outliers(episodes)
        valid = [episode for episode in episodes if episode["status"] == "valid"]
        excluded = [episode for episode in episodes if episode["status"] == "excluded"]
        lengths = [episode["steps"] for episode in valid]
        declared_total = data.attrs.get("total")
        if hasattr(declared_total, "item"):
            declared_total = declared_total.item()
        actual_total = sum(episode["steps"] for episode in episodes)
        if declared_total != actual_total:
            raise ValueError(f"total transition count mismatch: {task_key}")
        state_counts: dict[str, int] = {}
        for episode in valid:
            state = episode["init_state_sha256"]
            state_counts[state] = state_counts.get(state, 0) + 1
        duplicate_states = sum(count - 1 for count in state_counts.values())
        if duplicate_states:
            for episode in valid:
                if state_counts[episode["init_state_sha256"]] > 1:
                    episode["warnings"].append("duplicate_init_state")
                    episode["warnings"] = sorted(set(episode["warnings"]))
        return {
            "task": task_key,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest,
            "scene_id": scene_id,
            "language": language,
            "episodes": episodes,
            "summary": {
                "source_episodes": len(episodes),
                "valid_episodes": len(valid),
                "excluded_episodes": len(excluded),
                "transitions": sum(lengths),
                "min_steps": min(lengths) if lengths else None,
                "median_steps": statistics.median(lengths) if lengths else None,
                "mean_steps": statistics.fmean(lengths) if lengths else None,
                "max_steps": max(lengths) if lengths else None,
                "source_seeds_present": sum(e["source_seed"] is not None for e in valid),
                "duplicate_init_states": duplicate_states,
                "warnings": sum(len(e["warnings"]) for e in valid),
            },
        }


def _stable_group_order(task_key: str, group_key: str, split_seed: int) -> str:
    value = f"{split_seed}:{task_key}:{group_key}".encode()
    return hashlib.sha256(value).hexdigest()


def assign_splits(task_key: str, episodes: list[dict], split: dict) -> list[dict]:
    """Assign intact scene/seed groups to exact per-task split capacities."""

    valid = [episode for episode in episodes if episode["status"] == "valid"]
    expected = sum(split[name] for name in SPLITS)
    if len(valid) != expected:
        raise ValueError(f"{task_key} has {len(valid)} valid episodes; expected {expected}")
    grouped: dict[str, list[dict]] = {}
    for episode in valid:
        grouped.setdefault(episode["split_group"], []).append(episode)
    groups = sorted(
        grouped.values(),
        key=lambda group: _stable_group_order(task_key, group[0]["split_group"], split["seed"]),
    )
    remaining = {name: split[name] for name in SPLITS}
    rows: list[dict] = []
    for group in groups:
        size = len(group)
        choices = [name for name in SPLITS if remaining[name] >= size]
        if not choices:
            raise ValueError(f"cannot split {task_key} without leaking a scene/seed group")
        destination = max(choices, key=lambda name: (remaining[name], -SPLITS.index(name)))
        remaining[destination] -= size
        for episode in sorted(group, key=lambda item: item["episode_id"]):
            rows.append(
                {
                    "task": task_key,
                    "episode_id": episode["episode_id"],
                    "split": destination,
                    "steps": episode["steps"],
                    "source_seed": episode["source_seed"],
                    "scene_id": episode["scene_id"],
                    "scene_key": episode["scene_key"],
                    "split_group": episode["split_group"],
                }
            )
    if any(remaining.values()):
        raise ValueError(f"could not satisfy exact split counts for {task_key}: {remaining}")
    return sorted(rows, key=lambda row: (SPLITS.index(row["split"]), row["episode_id"]))


def _split_summary(rows: list[dict]) -> dict:
    summary = {}
    for task in TASKS:
        summary[task] = {
            split: sum(row["task"] == task and row["split"] == split for row in rows)
            for split in SPLITS
        }
    return summary


def validate_split_manifest(manifest: dict, expected: dict | None = None) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported split manifest schema")
    rows = manifest.get("episodes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("empty split manifest")
    identities = [(row.get("task"), row.get("episode_id")) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate episode in split manifest")
    group_splits: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row.get("task") not in TASKS or row.get("split") not in SPLITS:
            raise ValueError("invalid task or split")
        group = (row["task"], row.get("split_group"))
        group_splits.setdefault(group, set()).add(row["split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise ValueError("scene/seed group leaks across splits")
    if manifest.get("counts") != _split_summary(rows):
        raise ValueError("split summary does not match episode rows")
    if expected is not None:
        for task in TASKS:
            if manifest["counts"][task] != {name: expected[name] for name in SPLITS}:
                raise ValueError(f"split count mismatch: {task}")


def render_data_card(report: dict) -> str:
    """Create a compact Markdown snapshot from an audit report."""

    lines = [
        "# LIBERO Demonstration Data Card",
        "",
        f"Source: `{report['source']['repository']}` at `{report['source']['revision']}`.",
        "",
        "## Audited subset",
        "",
        "| Task | Episodes | Transitions | Length min / median / max | Excluded |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in TASKS:
        item = report["tasks"][task]["summary"]
        lines.append(
            f"| `{task}` | {item['valid_episodes']} | {item['transitions']:,} | "
            f"{item['min_steps']} / {item['median_steps']:g} / {item['max_steps']} | "
            f"{item['excluded_episodes']} |"
        )
    lines.extend(
        [
            "",
            "Actions have shape `(T, 7)`. Both RGB streams have shape "
            "`(T, 128, 128, 3)` and use `uint8` values.",
            "",
            "## Split policy",
            "",
            f"The deterministic split seed is `{report['split_seed']}`. Each task contributes "
            "40 training, 5 validation, and 5 test demonstrations. The upstream files do not "
            "store generator seeds, so the split groups demonstrations by the scene identifier "
            "and SHA-256 of the initial simulator state. An identical initial scene cannot cross "
            "split boundaries.",
            "",
            "## Exclusion policy",
            "",
            "An episode is excluded when a required array is missing, empty, unreadable, "
            "non-finite, length-misaligned, or inconsistent with the 7D action and two-camera "
            "RGB contract. File size, SHA-256, BDDL identity, declared demonstration count, and "
            "declared transition total must also match the pinned source.",
            "",
            "Length outliers, a false final `done` value, and duplicate initial states are "
            "reported for review but are not excluded automatically.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_audit(config_path: Path, repo_root: Path | None = None) -> Path:
    config, _ = load_audit_config(config_path)
    root = (repo_root or Path.cwd()).resolve()
    dataset_root = (root / config["dataset_root"]).resolve()
    output_root = (root / config["output_dir"]).resolve()
    for name, value in (("dataset_root", dataset_root), ("output_dir", output_root)):
        if root not in value.parents:
            raise ValueError(f"resolved {name} escapes or equals the repository root")

    task_results = {}
    rows = []
    for task_key in TASKS:
        source_path = dataset_root / config["tasks"][task_key]["file"]
        task_result = inspect_demo_file(
            source_path, task_key, config["tasks"][task_key], config["expected"]
        )
        task_results[task_key] = task_result
        for row in assign_splits(task_key, task_result["episodes"], config["split"]):
            row["source_file"] = task_result["file"]
            rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"audit_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=False, exist_ok=False)
    split_manifest = {
        "schema_version": 1,
        "source": config["source"],
        "split_seed": config["split"]["seed"],
        "seed_provenance": "upstream seed absent; init-state SHA-256 used as fallback",
        "counts": _split_summary(rows),
        "episodes": rows,
    }
    validate_split_manifest(split_manifest, config["split"])
    split_path = run_dir / "split_manifest.json"
    write_json(split_path, split_manifest)

    summaries = [task_results[task]["summary"] for task in TASKS]
    report = {
        "schema_version": 1,
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": config["source"],
        "config": str(config_path.as_posix()),
        "config_sha256": file_hash(config_path),
        "split_seed": config["split"]["seed"],
        "split_manifest": split_path.name,
        "split_manifest_sha256": file_hash(split_path),
        "tasks": task_results,
        "summary": {
            "source_episodes": sum(item["source_episodes"] for item in summaries),
            "valid_episodes": sum(item["valid_episodes"] for item in summaries),
            "excluded_episodes": sum(item["excluded_episodes"] for item in summaries),
            "transitions": sum(item["transitions"] for item in summaries),
            "source_seeds_present": sum(item["source_seeds_present"] for item in summaries),
            "warnings": sum(item["warnings"] for item in summaries),
            "task_imbalance": max(item["valid_episodes"] for item in summaries)
            - min(item["valid_episodes"] for item in summaries),
        },
    }
    report_path = run_dir / "audit.json"
    write_json(report_path, report)
    (run_dir / "data_card.md").write_text(render_data_card(report), encoding="utf-8")
    return report_path


def verify_audit(report_path: Path, repo_root: Path | None = None) -> dict:
    root = (repo_root or Path.cwd()).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("audit report is not complete schema v1")
    config_path = (root / report["config"]).resolve()
    if root not in config_path.parents or file_hash(config_path) != report["config_sha256"]:
        raise ValueError("audit config changed")
    config, _ = load_audit_config(config_path)
    split_path = report_path.parent / report["split_manifest"]
    if file_hash(split_path) != report["split_manifest_sha256"]:
        raise ValueError("split manifest checksum mismatch")
    manifest = json.loads(split_path.read_text(encoding="utf-8"))
    validate_split_manifest(manifest, config["split"])
    dataset_root = (root / config["dataset_root"]).resolve()
    expected_rows = []
    for task_key in TASKS:
        item = config["tasks"][task_key]
        source = dataset_root / item["file"]
        current = inspect_demo_file(source, task_key, item, config["expected"])
        if current != report["tasks"].get(task_key):
            raise ValueError(f"source or audit result changed: {task_key}")
        for row in assign_splits(task_key, current["episodes"], config["split"]):
            row["source_file"] = current["file"]
            expected_rows.append(row)
    if manifest["episodes"] != expected_rows:
        raise ValueError("split manifest does not reproduce from audited sources")
    return {
        "status": "verified",
        "episodes": len(manifest["episodes"]),
        "tasks": len(manifest["counts"]),
        "split_manifest_sha256": report["split_manifest_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit pinned official LIBERO demonstrations")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, default=Path("configs/data_audit.toml"))
    audit = commands.add_parser("run")
    audit.add_argument("--config", type=Path, default=Path("configs/data_audit.toml"))
    verify = commands.add_parser("verify")
    verify.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            config, _ = load_audit_config(args.config)
            result = {
                "source": config["source"],
                "dataset_root": config["dataset_root"],
                "tasks": {
                    key: {
                        "file": config["tasks"][key]["file"],
                        "bytes": config["tasks"][key]["bytes"],
                    }
                    for key in TASKS
                },
                "split": config["split"],
            }
            print(json.dumps(result, indent=2))
        elif args.command == "run":
            print(run_audit(args.config))
        else:
            print(json.dumps(verify_audit(args.report), indent=2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"demo audit error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
