"""Convert audited LIBERO demonstrations into a deterministic training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tomllib
import unicodedata
import uuid
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from robust_vla_recovery.data.demo_audit import (
    SPLITS,
    TASKS,
    _split_summary,
    assign_splits,
    inspect_demo_file,
    load_audit_config,
    validate_split_manifest,
)
from robust_vla_recovery.data.rollout import file_hash, write_json

STATE_FIELDS = ("ee_states", "gripper_states", "joint_states")
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]", re.ASCII)
EPISODE_PATTERN = re.compile(r"demo_(\d+)")


def _safe_relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).drive:
        raise ValueError(f"{field} must be repository-relative")
    if ".." in value.replace("\\", "/").split("/"):
        raise ValueError(f"{field} must not escape the repository")
    return path


def _exact_keys(value: dict, expected: set[str], field: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"invalid {field} keys: missing={missing}, extra={extra}")


def load_conversion_config(path: str | Path) -> dict:
    """Load and validate the public Day 9 conversion contract."""

    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    _exact_keys(
        config,
        {
            "schema_version",
            "audit_config",
            "output_dir",
            "control_frequency",
            "state_fields",
            "compression",
            "tokenizer",
        },
        "config",
    )
    if config["schema_version"] != 1:
        raise ValueError("unsupported conversion schema")
    for field in ("audit_config", "output_dir"):
        config[field] = str(_safe_relative_path(config[field], field)).replace("\\", "/")
    frequency = config["control_frequency"]
    if type(frequency) is not int or frequency < 1 or 1_000_000_000 % frequency:
        raise ValueError("control_frequency must divide one billion exactly")
    if config["state_fields"] != list(STATE_FIELDS):
        raise ValueError(f"state_fields must be {list(STATE_FIELDS)}")

    compression = config["compression"]
    if not isinstance(compression, dict):
        raise ValueError("compression must be a table")
    _exact_keys(compression, {"codec", "level"}, "compression")
    if compression["codec"] != "gzip":
        raise ValueError("only lossless gzip compression is supported")
    if type(compression["level"]) is not int or not 0 <= compression["level"] <= 9:
        raise ValueError("compression.level must be between 0 and 9")

    tokenizer = config["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer must be a table")
    _exact_keys(tokenizer, {"name", "lowercase"}, "tokenizer")
    if tokenizer != {"name": "lowercase-wordpunct-v1", "lowercase": True}:
        raise ValueError("unsupported tokenizer contract")
    return config


def tokenize(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip() or "\x00" in text:
        raise ValueError("language must be non-empty text without NUL")
    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens = TOKEN_PATTERN.findall(normalized)
    if not tokens:
        raise ValueError("language did not produce any tokens")
    return tokens


def build_vocabulary(
    languages: dict[str, str], rows: list[dict]
) -> tuple[list[str], dict[str, int]]:
    """Build a stable vocabulary from training instructions only."""

    training_tasks = {row["task"] for row in rows if row["split"] == "train"}
    words = sorted({word for task in training_tasks for word in tokenize(languages[task])})
    tokens = [*SPECIAL_TOKENS, *words]
    return tokens, {token: index for index, token in enumerate(tokens)}


def encode_language(text: str, vocabulary: dict[str, int]) -> list[int]:
    unknown = vocabulary["<unk>"]
    return [
        vocabulary["<bos>"],
        *(vocabulary.get(token, unknown) for token in tokenize(text)),
        vocabulary["<eos>"],
    ]


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(value: Any) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _resolve_under_root(root: Path, relative: str, field: str) -> Path:
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"resolved {field} escapes or equals the repository root")
    return path


def audit_sources(config: dict, root: Path) -> tuple[dict, dict[str, dict], list[dict]]:
    """Recreate the Day 8 audit and split rows from the pinned source files."""

    audit_path = _resolve_under_root(root, config["audit_config"], "audit_config")
    audit, _ = load_audit_config(audit_path)
    dataset_root = _resolve_under_root(root, audit["dataset_root"], "dataset_root")
    task_results: dict[str, dict] = {}
    rows: list[dict] = []
    for task in TASKS:
        item = audit["tasks"][task]
        result = inspect_demo_file(
            dataset_root / item["file"], task, item, audit["expected"]
        )
        task_results[task] = result
        for row in assign_splits(task, result["episodes"], audit["split"]):
            row["source_file"] = result["file"]
            rows.append(row)
    split_manifest = {
        "schema_version": 1,
        "counts": _split_summary(rows),
        "episodes": rows,
    }
    validate_split_manifest(split_manifest, audit["split"])
    return audit, task_results, rows


def _ordered_rows(rows: list[dict]) -> list[dict]:
    def episode_number(row: dict) -> int:
        match = EPISODE_PATTERN.fullmatch(row["episode_id"])
        if match is None:
            raise ValueError(f"invalid episode identifier: {row['episode_id']}")
        return int(match.group(1))

    return sorted(
        rows,
        key=lambda row: (
            SPLITS.index(row["split"]),
            TASKS.index(row["task"]),
            episode_number(row),
        ),
    )


def _dataset_kwargs(shape: tuple[int, ...], level: int) -> dict:
    leading = min(shape[0], 128)
    chunks = (1, *shape[1:]) if len(shape) == 4 else (leading, *shape[1:])
    return {
        "chunks": chunks,
        "compression": "gzip",
        "compression_opts": level,
        "shuffle": True,
        "fletcher32": True,
        "track_times": False,
    }


def _create_dataset(group: Any, name: str, value: Any, level: int) -> Any:
    return group.create_dataset(name, data=value, **_dataset_kwargs(value.shape, level))


def write_training_dataset(
    path: Path,
    dataset_root: Path,
    audit: dict,
    task_results: dict[str, dict],
    rows: list[dict],
    config: dict,
) -> dict:
    """Write one deterministic HDF5 file and return its deterministic manifest fields."""

    import h5py
    import numpy as np

    ordered = _ordered_rows(rows)
    languages = {task: task_results[task]["language"] for task in TASKS}
    vocabulary, token_to_id = build_vocabulary(languages, ordered)
    level = config["compression"]["level"]
    delta_ns = 1_000_000_000 // config["control_frequency"]
    lengths = np.asarray([row["steps"] for row in ordered], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lengths)))
    records: list[dict] = []

    with h5py.File(path, "w", libver="earliest", track_order=False) as output:
        output.attrs["schema_version"] = 1
        output.attrs["format"] = "rvla-policy-training-v1"
        output.attrs["control_frequency"] = config["control_frequency"]
        output.attrs["split_seed"] = audit["split"]["seed"]
        output.attrs["tokenizer"] = config["tokenizer"]["name"]
        output.attrs["vocabulary_json"] = json.dumps(
            vocabulary, ensure_ascii=False, separators=(",", ":")
        )
        output.attrs["state_fields_json"] = json.dumps(
            list(STATE_FIELDS), separators=(",", ":")
        )
        output.create_dataset("episode_lengths", data=lengths, track_times=False)
        output.create_dataset("episode_offsets", data=offsets, track_times=False)
        episodes = output.create_group("episodes", track_order=False)

        sources: dict[str, Any] = {}
        try:
            for index, row in enumerate(ordered):
                task = row["task"]
                source_file = row["source_file"]
                if source_file not in sources:
                    sources[source_file] = h5py.File(dataset_root / source_file, "r")
                source = sources[source_file][f"data/{row['episode_id']}"]
                steps = row["steps"]
                episode = episodes.create_group(f"{index:06d}", track_order=False)
                for name in (
                    "task",
                    "source_file",
                    "episode_id",
                    "split",
                    "scene_id",
                    "scene_key",
                    "split_group",
                ):
                    episode.attrs[name] = row[name]
                episode.attrs["source_seed"] = (
                    -1 if row["source_seed"] is None else row["source_seed"]
                )
                episode.attrs["steps"] = steps
                episode.attrs["language"] = languages[task]

                observations = episode.create_group("observation", track_order=False)
                agent = source["obs/agentview_rgb"][()]
                wrist = source["obs/eye_in_hand_rgb"][()]
                state = np.concatenate(
                    [source[f"obs/{name}"][()] for name in STATE_FIELDS], axis=1
                ).astype(np.float32)
                action = source["actions"][()].astype(np.float32)
                frame_index = np.arange(steps, dtype=np.int64)
                timestamp_ns = frame_index * delta_ns
                language_index = np.zeros(steps, dtype=np.int32)
                _create_dataset(observations, "agent_image", agent, level)
                _create_dataset(observations, "wrist_image", wrist, level)
                _create_dataset(observations, "state", state, level)
                _create_dataset(observations, "frame_index", frame_index, level)
                _create_dataset(observations, "timestamp_ns", timestamp_ns, level)
                _create_dataset(observations, "language_index", language_index, level)
                _create_dataset(episode, "action", action, level)
                token_ids = np.asarray(
                    encode_language(languages[task], token_to_id), dtype=np.int32
                )
                episode.create_dataset(
                    "language_token_ids", data=token_ids, track_times=False
                )
                episode.create_dataset(
                    "language_attention_mask",
                    data=np.ones(token_ids.shape, dtype=np.uint8),
                    track_times=False,
                )
                records.append(
                    {
                        "index": index,
                        **row,
                        "language": languages[task],
                        "language_token_ids": token_ids.tolist(),
                        "global_offset": int(offsets[index]),
                    }
                )
        finally:
            for source in sources.values():
                source.close()

    source_files = {
        task: {
            "file": task_results[task]["file"],
            "bytes": task_results[task]["bytes"],
            "sha256": task_results[task]["sha256"],
        }
        for task in TASKS
    }
    return {
        "source": audit["source"],
        "split_seed": audit["split"]["seed"],
        "source_files": source_files,
        "vocabulary": vocabulary,
        "vocabulary_sha256": _json_hash(vocabulary),
        "episodes": records,
        "counts": _split_summary(records),
        "summary": {
            "episodes": len(records),
            "transitions": int(lengths.sum()),
            "state_dim": 15,
            "action_dim": audit["expected"]["action_dim"],
            "image_shape": audit["expected"]["image_shape"],
        },
    }


def validate_training_dataset(path: Path, manifest: dict) -> dict:
    """Reject broken alignment, boundaries, schemas, and split leakage."""

    import h5py
    import numpy as np

    rows = manifest["episodes"]
    split_manifest = {
        "schema_version": 1,
        "counts": manifest["counts"],
        "episodes": rows,
    }
    validate_split_manifest(split_manifest)
    delta_ns = 1_000_000_000 // manifest["format"]["control_frequency"]
    with h5py.File(path, "r") as h5:
        if h5.attrs.get("schema_version") != 1:
            raise ValueError("training dataset has an unsupported schema")
        if h5.attrs.get("format") != "rvla-policy-training-v1":
            raise ValueError("training dataset has an unexpected format")
        if h5.attrs.get("control_frequency") != manifest["format"]["control_frequency"]:
            raise ValueError("training dataset control frequency changed")
        if h5.attrs.get("split_seed") != manifest["split_seed"]:
            raise ValueError("training dataset split seed changed")
        if h5.attrs.get("tokenizer") != manifest["format"]["tokenizer"]["name"]:
            raise ValueError("training dataset tokenizer changed")
        if list(h5["episodes"]) != [f"{index:06d}" for index in range(len(rows))]:
            raise ValueError("training episode index is incomplete or unordered")
        lengths = h5["episode_lengths"][()]
        offsets = h5["episode_offsets"][()]
        if not np.array_equal(lengths, [row["steps"] for row in rows]):
            raise ValueError("episode lengths disagree with the manifest")
        expected_offsets = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(lengths, dtype=np.int64))
        )
        if not np.array_equal(offsets, expected_offsets):
            raise ValueError("episode offsets do not preserve boundaries")

        vocabulary = json.loads(h5.attrs["vocabulary_json"])
        if vocabulary != manifest["vocabulary"]:
            raise ValueError("training dataset vocabulary changed")
        if _json_hash(vocabulary) != manifest["vocabulary_sha256"]:
            raise ValueError("training vocabulary checksum mismatch")
        token_to_id = {token: index for index, token in enumerate(vocabulary)}
        for index, row in enumerate(rows):
            episode = h5[f"episodes/{index:06d}"]
            steps = row["steps"]
            observation = episode["observation"]
            for name in ("agent_image", "wrist_image"):
                dataset = observation[name]
                if dataset.shape != (steps, *manifest["summary"]["image_shape"]):
                    raise ValueError(f"invalid {name} shape at episode {index}")
                if dataset.dtype != np.uint8 or dataset.compression != "gzip":
                    raise ValueError(f"invalid {name} storage at episode {index}")
            state = observation["state"]
            action = episode["action"]
            if state.shape != (steps, 15) or state.dtype != np.float32:
                raise ValueError(f"invalid state array at episode {index}")
            if action.shape != (steps, 7) or action.dtype != np.float32:
                raise ValueError(f"invalid action array at episode {index}")
            if not np.isfinite(state[()]).all() or not np.isfinite(action[()]).all():
                raise ValueError(f"non-finite training values at episode {index}")
            frame_index = np.arange(steps, dtype=np.int64)
            if not np.array_equal(observation["frame_index"], frame_index):
                raise ValueError(f"frame index mismatch at episode {index}")
            if not np.array_equal(observation["timestamp_ns"], frame_index * delta_ns):
                raise ValueError(f"timestamp mismatch at episode {index}")
            if not np.array_equal(observation["language_index"], np.zeros(steps)):
                raise ValueError(f"language alignment mismatch at episode {index}")
            if episode.attrs["language"] != row["language"]:
                raise ValueError(f"language text mismatch at episode {index}")
            if encode_language(row["language"], token_to_id) != row["language_token_ids"]:
                raise ValueError(f"manifest language token mismatch at episode {index}")
            if episode["language_token_ids"][()].tolist() != row["language_token_ids"]:
                raise ValueError(f"language token mismatch at episode {index}")
            if not np.array_equal(
                episode["language_attention_mask"],
                np.ones(len(row["language_token_ids"]), dtype=np.uint8),
            ):
                raise ValueError(f"language mask mismatch at episode {index}")
            for name in ("task", "source_file", "episode_id", "split", "split_group"):
                if episode.attrs[name] != row[name]:
                    raise ValueError(f"episode metadata mismatch: {index}:{name}")
    return {
        "status": "verified",
        "episodes": len(rows),
        "transitions": manifest["summary"]["transitions"],
    }


def _base_manifest(
    root: Path,
    config_path: Path,
    config: dict,
    fields: dict,
    dataset_hash: str,
) -> dict:
    audit_path = _resolve_under_root(root, config["audit_config"], "audit_config")
    return {
        "schema_version": 1,
        "status": "complete",
        "conversion_config": config_path.relative_to(root).as_posix(),
        "conversion_config_sha256": file_hash(config_path),
        "audit_config": config["audit_config"],
        "audit_config_sha256": file_hash(audit_path),
        "format": {
            "name": "rvla-policy-training-v1",
            "dataset": "dataset.h5",
            "dataset_sha256": dataset_hash,
            "control_frequency": config["control_frequency"],
            "timestamp_unit": "nanoseconds",
            "observation_action_alignment": "observation[t] -> action[t]",
            "state_fields": list(STATE_FIELDS),
            "compression": config["compression"],
            "tokenizer": config["tokenizer"],
        },
        **fields,
    }


def run_conversion(config_path: Path, repo_root: Path | None = None) -> Path:
    root = (repo_root or Path.cwd()).resolve()
    config_path = _resolve_under_root(root, config_path.as_posix(), "conversion_config")
    config = load_conversion_config(config_path)
    output_root = _resolve_under_root(root, config["output_dir"], "output_dir")
    audit, task_results, rows = audit_sources(config, root)
    dataset_root = _resolve_under_root(root, audit["dataset_root"], "dataset_root")

    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"conversion_{stamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=False, exist_ok=False)
    partial = run_dir / "dataset.partial.h5"
    dataset = run_dir / "dataset.h5"
    try:
        fields = write_training_dataset(
            partial, dataset_root, audit, task_results, rows, config
        )
        partial.replace(dataset)
        manifest = _base_manifest(
            root, config_path, config, fields, file_hash(dataset)
        )
        validate_training_dataset(dataset, manifest)
        report = run_dir / "conversion.json"
        write_json(report, manifest)
        return report
    except Exception:
        shutil.rmtree(run_dir)
        raise


def verify_conversion(report_path: Path, repo_root: Path | None = None) -> dict:
    root = (repo_root or Path.cwd()).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("conversion report is not complete schema v1")
    config_path = _resolve_under_root(root, report["conversion_config"], "conversion_config")
    if file_hash(config_path) != report["conversion_config_sha256"]:
        raise ValueError("conversion config changed")
    config = load_conversion_config(config_path)
    audit_path = _resolve_under_root(root, report["audit_config"], "audit_config")
    if file_hash(audit_path) != report["audit_config_sha256"]:
        raise ValueError("audit config changed")
    dataset = report_path.parent / report["format"]["dataset"]
    if file_hash(dataset) != report["format"]["dataset_sha256"]:
        raise ValueError("training dataset checksum mismatch")
    result = validate_training_dataset(dataset, report)

    audit, task_results, rows = audit_sources(config, root)
    expected = {
        task: {
            "file": task_results[task]["file"],
            "bytes": task_results[task]["bytes"],
            "sha256": task_results[task]["sha256"],
        }
        for task in TASKS
    }
    if audit["source"] != report["source"] or expected != report["source_files"]:
        raise ValueError("source provenance changed")
    expected_rows = _ordered_rows(rows)
    languages = {task: task_results[task]["language"] for task in TASKS}
    vocabulary, token_to_id = build_vocabulary(languages, expected_rows)
    if vocabulary != report["vocabulary"]:
        raise ValueError("training vocabulary does not reproduce from training instructions")
    offset = 0
    expected_records = []
    for index, row in enumerate(expected_rows):
        expected_records.append(
            {
                "index": index,
                **row,
                "language": languages[row["task"]],
                "language_token_ids": encode_language(languages[row["task"]], token_to_id),
                "global_offset": offset,
            }
        )
        offset += row["steps"]
    if expected_records != report["episodes"]:
        raise ValueError("conversion rows do not reproduce from audited sources")
    result["dataset_sha256"] = report["format"]["dataset_sha256"]
    return result


def select_sample_records(records: list[dict], count: int) -> list[dict]:
    if type(count) is not int or count < 1 or count > len(records):
        raise ValueError("sample count must be between one and the episode count")
    tasks = [task for task in TASKS if any(row["task"] == task for row in records)]
    splits = [split for split in SPLITS if any(row["split"] == split for row in records)]
    pools = {
        (task, split): sorted(
            (row for row in records if row["task"] == task and row["split"] == split),
            key=lambda row: row["episode_id"],
        )
        for task in tasks
        for split in splits
    }
    selected: list[dict] = []
    used: set[tuple[str, str]] = set()
    cycle = max(1, len(tasks) * len(splits))
    for index in range(cycle):
        key = (tasks[index % len(tasks)], splits[index % len(splits)])
        if key not in used and pools[key]:
            selected.append(pools[key].pop(0))
            used.add(key)
        if len(selected) == count:
            return selected
    remainder = sorted(
        (row for pool in pools.values() for row in pool),
        key=lambda row: (row["index"], row["task"], row["episode_id"]),
    )
    return [*selected, *remainder[: count - len(selected)]]


def compare_samples(
    report_path: Path, count: int = 10, repo_root: Path | None = None
) -> Path:
    import h5py
    import numpy as np

    root = (repo_root or Path.cwd()).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = load_conversion_config(root / report["conversion_config"])
    audit, _ = load_audit_config(root / config["audit_config"])
    dataset_root = root / audit["dataset_root"]
    selected = select_sample_records(report["episodes"], count)
    converted_path = report_path.parent / report["format"]["dataset"]
    comparisons: list[dict] = []
    with h5py.File(converted_path, "r") as converted:
        sources: dict[str, Any] = {}
        try:
            for row in selected:
                source_file = row["source_file"]
                if source_file not in sources:
                    sources[source_file] = h5py.File(dataset_root / source_file, "r")
                source = sources[source_file][f"data/{row['episode_id']}"]
                episode = converted[f"episodes/{row['index']:06d}"]
                problem = json.loads(str(sources[source_file]["data"].attrs["problem_info"]))
                source_language = problem["language_instruction"]
                source_state = np.concatenate(
                    [source[f"obs/{name}"][()] for name in STATE_FIELDS], axis=1
                ).astype(np.float32)
                pairs = {
                    "agent_image": (
                        source["obs/agentview_rgb"][()],
                        episode["observation/agent_image"][()],
                    ),
                    "wrist_image": (
                        source["obs/eye_in_hand_rgb"][()],
                        episode["observation/wrist_image"][()],
                    ),
                    "state": (source_state, episode["observation/state"][()]),
                    "action": (
                        source["actions"][()].astype(np.float32),
                        episode["action"][()],
                    ),
                }
                arrays = {}
                for name, (before, after) in pairs.items():
                    equal = bool(np.array_equal(before, after))
                    arrays[name] = {
                        "equal": equal,
                        "source_sha256": _array_hash(before),
                        "converted_sha256": _array_hash(after),
                    }
                    if not equal:
                        identity = f"{row['task']}:{row['episode_id']}:{name}"
                        raise ValueError(f"sample differs: {identity}")
                frame = row["steps"] // 2
                vocabulary = {
                    token: index for index, token in enumerate(report["vocabulary"])
                }
                expected_tokens = encode_language(source_language, vocabulary)
                alignment = {
                    "language_text": (
                        source_language == row["language"] == episode.attrs["language"]
                    ),
                    "language_tokens": (
                        expected_tokens
                        == row["language_token_ids"]
                        == episode["language_token_ids"][()].tolist()
                    ),
                    "frame_index": int(episode["observation/frame_index"][frame]) == frame,
                    "timestamp_ns": int(episode["observation/timestamp_ns"][frame])
                    == frame * (1_000_000_000 // report["format"]["control_frequency"]),
                    "language_index": int(episode["observation/language_index"][frame]) == 0,
                }
                if not all(alignment.values()):
                    identity = f"{row['task']}:{row['episode_id']}"
                    raise ValueError(f"sample alignment differs: {identity}")
                comparisons.append(
                    {
                        "task": row["task"],
                        "episode_id": row["episode_id"],
                        "split": row["split"],
                        "converted_index": row["index"],
                        "review_frame": frame,
                        "language": row["language"],
                        "language_token_ids": row["language_token_ids"],
                        "timestamp_ns": int(episode["observation/timestamp_ns"][frame]),
                        "alignment": alignment,
                        "arrays": arrays,
                        "status": "match",
                    }
                )
        finally:
            for source in sources.values():
                source.close()
    result = {
        "schema_version": 1,
        "status": "match",
        "selection": "deterministic task/split rotation",
        "count": len(comparisons),
        "samples": comparisons,
    }
    output = report_path.parent / "sample_comparison.json"
    write_json(output, result)
    return output


def compare_runs(first: Path, second: Path) -> dict:
    first_report = json.loads(first.read_text(encoding="utf-8"))
    second_report = json.loads(second.read_text(encoding="utf-8"))
    first_dataset = first.parent / first_report["format"]["dataset"]
    second_dataset = second.parent / second_report["format"]["dataset"]
    result = {
        "status": "match",
        "dataset_sha256": file_hash(first_dataset),
        "report_sha256": file_hash(first),
    }
    if result["dataset_sha256"] != file_hash(second_dataset):
        raise ValueError("dataset checksums differ between runs")
    if result["report_sha256"] != file_hash(second):
        raise ValueError("conversion reports differ between runs")
    return result


def _plan(config_path: Path) -> dict:
    config = load_conversion_config(config_path)
    audit, _ = load_audit_config(config["audit_config"])
    return {
        "format": "rvla-policy-training-v1",
        "source": audit["source"],
        "tasks": list(TASKS),
        "split": audit["split"],
        "state_fields": config["state_fields"],
        "action_dim": audit["expected"]["action_dim"],
        "image_shape": audit["expected"]["image_shape"],
        "control_frequency": config["control_frequency"],
        "tokenizer": config["tokenizer"],
        "output_dir": config["output_dir"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert audited LIBERO demonstrations")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--config", type=Path, default=Path("configs/data_conversion.toml"))
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/data_conversion.toml"))
    verify = commands.add_parser("verify")
    verify.add_argument("report", type=Path)
    sample = commands.add_parser("sample-check")
    sample.add_argument("report", type=Path)
    sample.add_argument("--count", type=int, default=10)
    reproduce = commands.add_parser("repro-check")
    reproduce.add_argument("first", type=Path)
    reproduce.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result: Any = _plan(args.config)
        elif args.command == "run":
            result = run_conversion(args.config)
        elif args.command == "verify":
            result = verify_conversion(args.report)
        elif args.command == "sample-check":
            result = compare_samples(args.report, args.count)
        else:
            result = compare_runs(args.first, args.second)
        if isinstance(result, Path):
            print(result)
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"conversion error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
