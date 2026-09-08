"""Day 7 clean-condition baseline evaluation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
import tomllib
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from robust_vla_recovery.data.rollout import write_json
from robust_vla_recovery.envs.libero_smoke import prepare_libero_config, sha256_file
from robust_vla_recovery.envs.task_adapter import (
    RECOVERABLE_FAILURES, TaskAdapter, TaskSpec, load_task_catalog,
)
from robust_vla_recovery.policy import (
    LightweightBaselinePolicy, latency_summary, safe_repo_path,
)


DAY5_MEAN_EPISODE_BYTES = 64_119_799
WEEK2_SPLIT_PER_TASK = {"train": 40, "validation": 5, "test": 5}


@dataclass(frozen=True, slots=True)
class BaselineEvalConfig:
    catalog: Path
    output_dir: Path
    tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    init_state_indices: tuple[int, ...]
    instruction_indices: tuple[int, ...]
    policy_seed: int
    warmup_calls: int
    action_scale: tuple[float, ...]
    failure_videos: int


def load_eval_config(path: str | Path) -> tuple[BaselineEvalConfig, dict[str, TaskSpec]]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    expected = {"schema_version", "catalog", "output_dir", "tasks", "seeds",
                "init_state_indices", "instruction_indices", "policy_seed", "warmup_calls",
                "action_scale", "failure_videos"}
    if set(raw) != expected or raw["schema_version"] != 1:
        raise ValueError("invalid baseline evaluation schema")
    catalog_path = safe_repo_path(raw["catalog"], "catalog")
    output_dir = safe_repo_path(raw["output_dir"], "output_dir")
    catalog = load_task_catalog(catalog_path)
    tasks = raw["tasks"]
    if not isinstance(tasks, list) or set(tasks) != set(catalog) or len(tasks) != len(catalog):
        raise ValueError("evaluation must contain each catalog task exactly once")
    seeds = raw["seeds"]
    if (not isinstance(seeds, list) or len(seeds) != 10 or len(set(seeds)) != 10
            or not all(type(seed) is int and 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("evaluation requires ten unique uint32 seeds")
    for field in ("init_state_indices", "instruction_indices"):
        values = raw[field]
        if (not isinstance(values, list) or len(values) != len(seeds)
                or not all(type(value) is int and value >= 0 for value in values)):
            raise ValueError(f"invalid {field}")
    if any(index >= len(catalog[task].instructions)
           for task in tasks for index in raw["instruction_indices"]):
        raise ValueError("instruction index out of range")
    if type(raw["policy_seed"]) is not int or not 0 <= raw["policy_seed"] < 2**32:
        raise ValueError("policy_seed must be uint32")
    if type(raw["warmup_calls"]) is not int or not 1 <= raw["warmup_calls"] <= 100:
        raise ValueError("warmup_calls must be in [1, 100]")
    scale = raw["action_scale"]
    if (not isinstance(scale, list) or len(scale) != 7
            or not all(type(value) in (int, float) and 0 < value <= 1 for value in scale)):
        raise ValueError("action_scale must contain seven numbers in (0, 1]")
    if raw["failure_videos"] != 5:
        raise ValueError("Day 7 requires exactly five failure videos")
    config = BaselineEvalConfig(
        catalog=catalog_path, output_dir=output_dir, tasks=tuple(tasks), seeds=tuple(seeds),
        init_state_indices=tuple(raw["init_state_indices"]),
        instruction_indices=tuple(raw["instruction_indices"]), policy_seed=raw["policy_seed"],
        warmup_calls=raw["warmup_calls"], action_scale=tuple(float(x) for x in scale),
        failure_videos=raw["failure_videos"],
    )
    return config, catalog


def evaluation_requests(config: BaselineEvalConfig) -> list[dict]:
    requests = []
    for task in config.tasks:
        for seed, init, instruction in zip(config.seeds, config.init_state_indices,
                                           config.instruction_indices, strict=True):
            requests.append({
                "episode_id": f"{task}_seed{seed}_init{init}_lang{instruction}",
                "task": task, "seed": seed, "init_state_index": init,
                "instruction_index": instruction,
            })
    return requests


def aggregate_results(tasks: tuple[str, ...] | list[str], results: list[dict]) -> dict:
    def summarize(records: list[dict]) -> dict:
        completed = [record for record in records if record["status"] == "complete"]
        successes = sum(bool(record["success"]) for record in completed)
        calls = sum(record["inference"]["calls"] for record in completed)
        weighted_latency = sum(record["inference"]["mean_ms"]
                               * record["inference"]["calls"] for record in completed)
        return {
            "episodes": len(records), "completed": len(completed), "successes": successes,
            "success_rate": successes / len(completed) if completed else None,
            "mean_episode_seconds": (
                sum(record["episode_seconds"] for record in completed) / len(completed)
                if completed else None
            ),
            "mean_inference_ms": weighted_latency / calls if calls else None,
            "deadline_misses": sum(record["inference"]["deadline_misses"]
                                   for record in completed),
            "peak_gpu_allocated_bytes": max(
                (record["gpu"]["peak_allocated_bytes"] for record in completed), default=0
            ),
            "peak_gpu_reserved_bytes": max(
                (record["gpu"]["peak_reserved_bytes"] for record in completed), default=0
            ),
        }

    return {"overall": summarize(results),
            "by_task": {task: summarize([r for r in results if r["task"] == task])
                        for task in tasks}}


def week2_data_plan(task_count: int) -> dict:
    """Return the fixed Week 2 target and a measured storage budget."""
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    episodes_per_task = sum(WEEK2_SPLIT_PER_TASK.values())
    episodes = task_count * episodes_per_task
    estimated_bytes = episodes * DAY5_MEAN_EPISODE_BYTES
    storage_budget_bytes = (estimated_bytes * 6 + 4) // 5
    return {
        "target_valid_episodes_per_task": episodes_per_task,
        "split_per_task": WEEK2_SPLIT_PER_TASK.copy(),
        "tasks": task_count,
        "target_valid_episodes": episodes,
        "target_transitions": episodes * 400,
        "day5_measured_mean_episode_bytes": DAY5_MEAN_EPISODE_BYTES,
        "estimated_dataset_bytes": estimated_bytes,
        "storage_budget_bytes_with_20_percent_headroom": storage_budget_bytes,
        "source_validity_rule": (
            "Never synthesize, duplicate, or move an episode across splits to meet the target; "
            "record any source shortage or invalid episode as an exclusion."
        ),
    }


def select_failure_ids(
    tasks: tuple[str, ...] | list[str], results: list[dict], count: int,
) -> list[str]:
    failures = [r for r in results if r["status"] == "complete" and not r["success"]]
    selected = []
    for task in tasks:
        match = next((r for r in failures if r["task"] == task), None)
        if match is not None:
            selected.append(match["episode_id"])
    for result in failures:
        if len(selected) == count:
            break
        if result["episode_id"] not in selected:
            selected.append(result["episode_id"])
    return selected[:count]


def _gpu_metrics(*, reset: bool = False) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"available": False, "device": None, "total_bytes": 0,
                "peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
    torch.cuda.synchronize()
    if reset:
        torch.cuda.reset_peak_memory_stats()
    properties = torch.cuda.get_device_properties(0)
    return {"available": True, "device": properties.name, "total_bytes": properties.total_memory,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _video_frame(observation: dict):
    import numpy as np

    left = observation["pixels"]["image"][::-1, ::-1]
    right = observation["pixels"]["image2"][::-1, ::-1]
    return np.ascontiguousarray(np.concatenate((left, right), axis=1))


def _run_episode(root: Path, request: dict, spec: TaskSpec, policy: LightweightBaselinePolicy,
                 warmup_calls: int) -> dict:
    import imageio.v2 as imageio
    import numpy as np

    from robust_vla_recovery.envs.libero_tasks import LiberoTaskBackend

    started = time.perf_counter()
    backend = LiberoTaskBackend(spec)
    adapter = TaskAdapter(spec, backend, instruction_index=request["instruction_index"],
                          init_state_index=request["init_state_index"])
    partial = root / f"{request['episode_id']}.partial.mp4"
    video = root / f"{request['episode_id']}.mp4"
    latencies = []
    actions = []
    primary_error: BaseException | None = None
    result = None
    try:
        observation, _ = adapter.reset(seed=request["seed"])
        _gpu_metrics(reset=True)
        for _ in range(warmup_calls):
            policy.predict(observation, adapter.task_description)
        episode_started = time.perf_counter()
        reward_sum = 0.0
        info = adapter.last_info
        with imageio.get_writer(partial, fps=spec.control_frequency, codec="libx264",
                                macro_block_size=None) as writer:
            writer.append_data(_video_frame(observation))
            for _ in range(spec.horizon):
                inference_started = time.perf_counter_ns()
                action = policy.predict(observation, adapter.task_description)
                latencies.append(time.perf_counter_ns() - inference_started)
                actions.append(action.copy())
                observation, reward, terminated, truncated, info = adapter.step(action)
                reward_sum += reward
                writer.append_data(_video_frame(observation))
                if terminated or truncated:
                    break
        partial.replace(video)
        action_array = np.stack(actions)
        result = {
            **request, "status": "complete", "steps": len(actions),
            "frames": len(actions) + 1, "success": bool(info["is_success"]),
            "stable_success": bool(info["stable_success"]), "return": reward_sum,
            "termination_reason": info["termination_reason"],
            "rollout_seconds": time.perf_counter() - episode_started,
            "inference": latency_summary(latencies, spec.control_frequency),
            "action_min": action_array.min(axis=0).tolist(),
            "action_max": action_array.max(axis=0).tolist(),
            "gpu": _gpu_metrics(), "video": video.name,
        }
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            adapter.close()
        except Exception as exc:
            if primary_error is None:
                raise
            primary_error.add_note(f"adapter cleanup also failed: {exc!r}")
        if primary_error is not None:
            partial.unlink(missing_ok=True)
    result["episode_seconds"] = time.perf_counter() - started
    return result


def _config_json(config: BaselineEvalConfig) -> dict:
    value = asdict(config)
    value["catalog"] = str(config.catalog)
    value["output_dir"] = str(config.output_dir)
    value["tasks"] = list(config.tasks)
    value["seeds"] = list(config.seeds)
    value["init_state_indices"] = list(config.init_state_indices)
    value["instruction_indices"] = list(config.instruction_indices)
    value["action_scale"] = list(config.action_scale)
    return value


def run_evaluation(config_path: str | Path) -> Path:
    config, catalog = load_eval_config(config_path)
    if platform.system() != "Linux":
        raise ValueError("baseline evaluation requires Linux/WSL2")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root = config.output_dir.resolve() / f"eval_{stamp}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    libero_config = prepare_libero_config(root)
    requests = evaluation_requests(config)
    policy = LightweightBaselinePolicy(seed=config.policy_seed, action_scale=config.action_scale)
    report = {
        "schema_version": 1, "status": "running", "config": _config_json(config),
        "config_sha256": sha256_file(config_path), "catalog_sha256": sha256_file(config.catalog),
        "policy": policy.contract(), "requests": requests, "results": [],
        "started_at_utc": datetime.now(UTC).isoformat(),
        "week2_data_plan": week2_data_plan(len(config.tasks)),
        "runtime": {"python": platform.python_version(),
                    "numpy": importlib.metadata.version("numpy"),
                    "torch": importlib.metadata.version("torch")},
    }
    report_path = root / "results.json"
    write_json(report_path, report)
    batch_started = time.perf_counter()
    for number, request in enumerate(requests, 1):
        print(f"[{number}/{len(requests)}] {request['episode_id']}", flush=True)
        try:
            result = _run_episode(root, request, catalog[request["task"]], policy,
                                  config.warmup_calls)
            print(f"  success={result['success']} steps={result['steps']} "
                  f"time={result['episode_seconds']:.2f}s", flush=True)
        except Exception as exc:
            result = {**request, "status": "error", "error": repr(exc), "video": None}
            print(f"  error={exc}", flush=True)
        report["results"].append(result)
        write_json(report_path, report)

    selected_ids = select_failure_ids(config.tasks, report["results"], config.failure_videos)
    selected = []
    for result in report["results"]:
        video_name = result.get("video")
        if not video_name:
            continue
        video_path = root / video_name
        if result["episode_id"] in selected_ids:
            selected.append({"episode_id": result["episode_id"], "video": video_name,
                             "sha256": sha256_file(video_path), "review": {"status": "pending"}})
        else:
            video_path.unlink()
            result["video"] = None
    libero_config.unlink(missing_ok=True)
    libero_config.parent.rmdir()
    report["summary"] = aggregate_results(config.tasks, report["results"])
    report["selected_failures"] = selected
    report["batch_seconds"] = time.perf_counter() - batch_started
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    complete_runs = all(result["status"] == "complete" for result in report["results"])
    report["status"] = ("review_pending" if complete_runs and len(selected) == config.failure_videos
                        else "failed")
    write_json(report_path, report)
    if report["status"] == "failed":
        raise RuntimeError("baseline evaluation failed; inspect results.json")
    return report_path


def record_evaluation_review(path: str | Path, episode_id: str, candidates: list[str],
                             frames: list[int], note: str) -> None:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.setdefault("week2_data_plan", week2_data_plan(len(report["config"]["tasks"])))
    if report.get("status") not in {"review_pending", "complete"}:
        raise ValueError("evaluation is not ready for review")
    entry = next((item for item in report["selected_failures"]
                  if item["episode_id"] == episode_id), None)
    if entry is None:
        raise ValueError("episode is not a selected failure")
    invalid = set(candidates) - RECOVERABLE_FAILURES
    if not candidates or invalid:
        raise ValueError(f"invalid failure candidates: {sorted(invalid)}")
    result = next(item for item in report["results"] if item["episode_id"] == episode_id)
    if not frames or any(type(frame) is not int or not 0 <= frame <= result["steps"]
                         for frame in frames):
        raise ValueError("review frames must be within the episode")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("review note is required")
    entry["review"] = {"status": "reviewed", "candidate_types": sorted(set(candidates)),
                       "frames": sorted(set(frames)), "note": note.strip(),
                       "candidate_only": True}
    if all(item["review"]["status"] == "reviewed" for item in report["selected_failures"]):
        report["status"] = "complete"
    write_json(report_path, report)


def verify_evaluation(path: str | Path) -> dict:
    report_path = Path(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "complete":
        raise ValueError("evaluation is not complete")
    config = report["config"]
    expected = []
    for task in config["tasks"]:
        for seed, init, instruction in zip(config["seeds"], config["init_state_indices"],
                                           config["instruction_indices"], strict=True):
            expected.append({"episode_id": f"{task}_seed{seed}_init{init}_lang{instruction}",
                             "task": task, "seed": seed, "init_state_index": init,
                             "instruction_index": instruction})
    if report["requests"] != expected or len(report["results"]) != len(expected):
        raise ValueError("evaluation matrix mismatch")
    result_matrix = [
        {key: result[key] for key in (
            "episode_id", "task", "seed", "init_state_index", "instruction_index"
        )}
        for result in report["results"]
    ]
    if result_matrix != expected:
        raise ValueError("episode results do not match the evaluation matrix")
    if any(result["status"] != "complete" for result in report["results"]):
        raise ValueError("evaluation contains failed episodes")
    if aggregate_results(config["tasks"], report["results"]) != report["summary"]:
        raise ValueError("summary does not match episode results")
    if report.get("week2_data_plan") != week2_data_plan(len(config["tasks"])):
        raise ValueError("Week 2 data plan mismatch")
    selected = report["selected_failures"]
    if len(selected) != 5 or any(item["review"]["status"] != "reviewed" for item in selected):
        raise ValueError("five reviewed failures are required")
    expected_selected = select_failure_ids(config["tasks"], report["results"], 5)
    selected_ids = [item["episode_id"] for item in selected]
    if len(set(selected_ids)) != len(selected_ids) or set(selected_ids) != set(expected_selected):
        raise ValueError("selected failures do not match the selection rule")
    selected_names = set()
    for item in selected:
        result = next(r for r in report["results"] if r["episode_id"] == item["episode_id"])
        if result["success"] or item["video"] != result["video"]:
            raise ValueError("selected failure does not match episode result")
        review = item["review"]
        candidates = review.get("candidate_types")
        frames = review.get("frames")
        if (not isinstance(candidates, list) or not candidates
                or set(candidates) - RECOVERABLE_FAILURES
                or candidates != sorted(set(candidates))):
            raise ValueError("invalid failure review candidates")
        if (not isinstance(frames, list) or not frames
                or frames != sorted(set(frames))
                or any(type(frame) is not int or not 0 <= frame <= result["steps"]
                       for frame in frames)
                or review.get("candidate_only") is not True
                or not isinstance(review.get("note"), str) or not review["note"].strip()):
            raise ValueError("invalid failure review")
        if Path(item["video"]).name != item["video"]:
            raise ValueError("failure video must be a filename")
        video_path = report_path.parent / item["video"]
        if sha256_file(video_path) != item["sha256"]:
            raise ValueError("failure video checksum mismatch")
        selected_names.add(item["video"])
    if {video.name for video in report_path.parent.glob("*.mp4")} != selected_names:
        raise ValueError("unexpected or missing evaluation video")
    return {"status": "verified", "episodes": len(report["results"]),
            "selected_failure_videos": len(selected), "summary": report["summary"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run and review the Day 7 baseline evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--config", type=Path, default=Path("configs/baseline_eval.toml"))
    run.add_argument("--dry-run", action="store_true")
    review = commands.add_parser("review")
    review.add_argument("results", type=Path)
    review.add_argument("--episode", required=True)
    review.add_argument("--candidate", action="append", required=True,
                        choices=sorted(RECOVERABLE_FAILURES))
    review.add_argument("--frame", action="append", type=int, required=True)
    review.add_argument("--note", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("results", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            config, _ = load_eval_config(args.config)
            if args.dry_run:
                print(json.dumps({"episodes": len(evaluation_requests(config)),
                                  "config": _config_json(config)}, indent=2))
            else:
                print(run_evaluation(args.config))
        elif args.command == "review":
            record_evaluation_review(args.results, args.episode, args.candidate,
                                     args.frame, args.note)
        else:
            print(json.dumps(verify_evaluation(args.results), indent=2))
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"baseline evaluation error: {exc}\n")
    return 0
