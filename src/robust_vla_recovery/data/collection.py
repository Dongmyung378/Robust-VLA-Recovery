"""Day 5 collection manifest, storage estimate, and standalone replay CLI."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import time
import tomllib
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

from robust_vla_recovery.data.rollout import (
    EpisodeWriter, export_video, file_hash, verify_episode, write_json,
)
from robust_vla_recovery.envs.libero_smoke import prepare_libero_config
from robust_vla_recovery.envs.task_adapter import TaskAdapter, load_task_catalog


def load_collection(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    keys = {"schema_version", "catalog", "output_dir", "tasks", "seeds", "init_state_indices",
            "instruction_indices", "width", "height", "gzip_level", "perturbation"}
    if set(config) != keys or type(config["schema_version"]) is not int:
        raise ValueError("invalid collection schema")
    if config["schema_version"] != 1:
        raise ValueError("unsupported collection schema")
    for field in ("catalog", "output_dir"):
        value = config[field]
        if (not isinstance(value, str) or not value.strip() or Path(value).is_absolute()
                or PureWindowsPath(value).drive or ".." in value.replace("\\", "/").split("/")):
            raise ValueError(f"{field} must be a safe repository-relative path")
        resolved = Path(value).resolve()
        if Path.cwd().resolve() not in resolved.parents:
            raise ValueError(f"{field} escapes repository or equals its root")
    catalog = load_task_catalog(config["catalog"])
    if (not isinstance(config["tasks"], list) or not config["tasks"]
            or not all(isinstance(k, str) and k in catalog for k in config["tasks"])
            or len(set(config["tasks"])) != len(config["tasks"])):
        raise ValueError("invalid or duplicate task list")
    for field in ("seeds", "init_state_indices", "instruction_indices"):
        items = config[field]
        if (not isinstance(items, list) or not items
                or not all(type(x) is int and x >= 0 for x in items)):
            raise ValueError(f"invalid {field}")
    if len(set(config["seeds"])) != len(config["seeds"]) or max(config["seeds"]) >= 2**32:
        raise ValueError("seeds must be unique uint32 values")
    if any(len(config[f]) != len(config["seeds"])
           for f in ("init_state_indices", "instruction_indices")):
        raise ValueError("seed, initial-state and instruction lists must have equal lengths")
    if any(max(config["instruction_indices"]) >= len(catalog[k].instructions)
           for k in config["tasks"]):
        raise ValueError("instruction index out of range")
    for key in ("width", "height"):
        if type(config[key]) is not int or not 16 <= config[key] <= 1024:
            raise ValueError(f"invalid {key}")
    if type(config["gzip_level"]) is not int or not 0 <= config["gzip_level"] <= 9:
        raise ValueError("invalid gzip_level")
    if config["perturbation"] != {"enabled": False, "template_id": "clean-v1"}:
        raise ValueError("Day 5 collector only supports the explicitly clean-v1 condition")
    return config, catalog


def storage_estimate(config: dict, catalog: dict) -> dict:
    episodes = len(config["tasks"]) * len(config["seeds"])
    observations = sum(catalog[k].horizon + 1 for k in config["tasks"]) * len(config["seeds"])
    raw_rgb = observations * 2 * config["width"] * config["height"] * 3
    # Budget twice the uncompressed RGB to cover other arrays, metadata and overhead.
    return {"episodes": episodes, "observations": observations, "rgb_frames": observations * 2,
            "raw_rgb_bytes": raw_rgb, "required_free_bytes": raw_rgb * 2 + 100_000_000}


def collection_requests(config: dict) -> list[dict]:
    requests = []
    for key in config["tasks"]:
        for seed, init, instruction in zip(config["seeds"], config["init_state_indices"],
                                           config["instruction_indices"], strict=True):
            requests.append({"episode_id": f"{key}_seed{seed}_init{init}_lang{instruction}",
                             "task_key": key, "seed": seed, "init_state_index": init,
                             "instruction_index": instruction})
    return requests


def verify_batch(path: Path) -> dict:
    batch = json.loads(path.read_text(encoding="utf-8"))
    if batch.get("status") != "complete" or batch.get("schema_version") != 1:
        raise ValueError("batch is not complete")
    results = batch["results"]
    requests = batch["requests"]
    if requests != collection_requests(batch["config"]):
        raise ValueError("requests differ from the configured task/seed matrix")
    if len(results) != len(requests) or not requests:
        raise ValueError("batch count mismatch")
    ids = [request["episode_id"] for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode ID")
    verified = []
    for request, result in zip(requests, results, strict=True):
        name = request["episode_id"]
        if (not isinstance(name, str) or Path(name).name != name
                or name in {".", ".."} or "\\" in name or ":" in name):
            raise ValueError("unsafe episode ID")
        if result["status"] != "verified" or result["episode_id"] != name:
            raise ValueError("failed or mismatched batch result")
        metadata_path = path.parent / name / "metadata.json"
        if path.parent.resolve() not in metadata_path.resolve().parents:
            raise ValueError("episode path escapes batch")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(metadata["provenance"].get(k) != v for k, v in request.items()):
            raise ValueError("episode identity does not match requested task/seed/scene/language")
        if metadata["provenance"]["perturbation"] != batch["config"]["perturbation"]:
            raise ValueError("perturbation provenance mismatch")
        original_hash = result.get("sha256")
        if (not isinstance(original_hash, str) or len(original_hash) != 64
                or any(c not in "0123456789abcdef" for c in original_hash)):
            raise ValueError(f"missing or invalid original collection checksum: {name}")
        episode_result = verify_episode(metadata_path)
        if episode_result["sha256"] != original_hash:
            raise ValueError(f"original collection checksum mismatch: {name}")
        verified.append(episode_result)
        print(f"Replayed {name}: {verified[-1]['rgb_frames']} lossless RGB frames", flush=True)
    return {"status": "verified", "episodes": len(verified),
            "steps": sum(item["steps"] for item in verified),
            "rgb_frames": sum(item["rgb_frames"] for item in verified),
            "raw_rgb_bytes": sum(item["raw_rgb_bytes"] for item in verified),
            "stored_bytes": sum(item["stored_bytes"] for item in verified), "results": verified}


def collect(config_path: Path) -> Path:
    config, catalog = load_collection(config_path)
    if platform.system() != "Linux":
        raise ValueError("collection requires Linux/WSL2")
    estimate = storage_estimate(config, catalog)
    free = shutil.disk_usage(Path.cwd()).free
    if free < estimate["required_free_bytes"]:
        raise OSError(f"insufficient disk: need {estimate['required_free_bytes']}, have {free}")
    parent = Path(config["output_dir"]).resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = parent / f"batch_{stamp}_{uuid.uuid4().hex[:8]}"
    root.mkdir(parents=True, exist_ok=False)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    prepare_libero_config(parent)
    import numpy as np

    from robust_vla_recovery.envs.libero_tasks import LiberoTaskBackend

    requests = collection_requests(config)
    versions = {key: importlib.metadata.version(key)
                for key in ("numpy", "h5py", "hf-libero", "lerobot", "mujoco", "robosuite")}
    batch = {"schema_version": 1, "status": "running", "config": config,
             "config_sha256": file_hash(config_path),
             "catalog_sha256": file_hash(Path(config["catalog"])),
             "requests": requests, "results": [], "storage_estimate": estimate,
             "free_bytes_before": free, "versions": versions}
    batch_path = root / "batch.json"
    write_json(batch_path, batch)
    for number, request in enumerate(requests, 1):
        adapter = None
        result = None
        started = time.perf_counter()
        try:
            spec = catalog[request["task_key"]]
            backend = LiberoTaskBackend(spec, width=config["width"], height=config["height"])
            adapter = TaskAdapter(spec, backend, instruction_index=request["instruction_index"],
                                  init_state_index=request["init_state_index"])
            obs, _ = adapter.reset(seed=request["seed"])
            provenance = {**request, "task_name": spec.name, "libero_task_id": backend.task_id,
                          "suite": spec.suite, "horizon": spec.horizon,
                          "perturbation": config["perturbation"], "policy": "fixed_noop",
                          "task_spec": asdict(spec), "bddl_sha256": backend.bddl_sha256,
                          "catalog_sha256": batch["catalog_sha256"], "versions": versions,
                          "python": platform.python_version(),
                          "rgb_orientation": "raw LeRobot observations; no flip/resize",
                          "timestamp_origin": "post-reset-and-settle simulation time = 0"}
            print(f"[{number}/{len(requests)}] Collecting {request['episode_id']}", flush=True)
            with EpisodeWriter(root / request["episode_id"], provenance, obs,
                               adapter.task_description, control_frequency=spec.control_frequency,
                               gzip_level=config["gzip_level"]) as writer:
                for step in range(spec.horizon):
                    action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                    obs, reward, terminated, truncated, info = adapter.step(action)
                    writer.append(step, action, reward, obs, info["language"],
                                  terminated=terminated, truncated=truncated, info=info)
                    if terminated or truncated:
                        break
                metadata_path = writer.finish(final_info=adapter.last_info, events=adapter.events)
            result = verify_episode(metadata_path)
            result["wall_seconds"] = time.perf_counter() - started
            batch["results"].append(result)
            print(f"Verified {request['episode_id']}: {result['steps']} steps, "
                  f"{result['stored_bytes'] / 2**20:.2f} MiB", flush=True)
        except Exception as exc:
            result = {
                "episode_id": request["episode_id"], "status": "error",
                "error": repr(exc), "events": adapter.events if adapter else [],
            }
            batch["results"].append(result)
            print(f"FAILED {request['episode_id']}: {exc}", flush=True)
        finally:
            try:
                if adapter is not None:
                    adapter.close()
            except Exception as exc:
                if result is None:
                    result = {"episode_id": request["episode_id"], "status": "error"}
                    batch["results"].append(result)
                # Preserve the primary error or verified artifact details separately.
                result.update(status="error", cleanup_error=repr(exc))
                print(f"CLEANUP FAILED {request['episode_id']}: {exc}", flush=True)
            finally:
                write_json(batch_path, batch)
    all_verified = all(r["status"] == "verified" for r in batch["results"])
    batch["status"] = "complete" if all_verified else "failed"
    write_json(batch_path, batch)
    print(f"Batch: {batch_path}", flush=True)
    if batch["status"] != "complete":
        raise RuntimeError("collection contains failed episodes; inspect batch.json")
    return batch_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect and replay lossless Day 5 rollouts")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect")
    run.add_argument("--config", type=Path, default=Path("configs/rollout.toml"))
    run.add_argument("--dry-run", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("batch", type=Path)
    replay = commands.add_parser("replay")
    replay.add_argument("metadata", type=Path)
    replay.add_argument("--video", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "collect":
            if args.dry_run:
                config, catalog = load_collection(args.config)
                print(json.dumps(storage_estimate(config, catalog), indent=2))
            else:
                collect(args.config)
        elif args.command == "verify":
            result = verify_batch(args.batch)
            write_json(args.batch.parent / "verification.json", result)
            print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
        else:
            print(json.dumps(verify_episode(args.metadata), indent=2))
            if args.video:
                export_video(args.metadata, args.video)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, f"rollout error: {exc}\n")
    return 0
