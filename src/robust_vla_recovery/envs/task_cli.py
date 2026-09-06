"""Single CLI for listing and running all four frozen project tasks."""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from robust_vla_recovery.envs.libero_smoke import (
    LiberoSmokeConfig,
    _json_default,
    prepare_libero_config,
    resolve_artifact_root,
    run_libero_smoke,
    sha256_file,
    verify_episode_artifacts,
)
from robust_vla_recovery.envs.task_adapter import TaskAdapter, load_task_catalog


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List or run the four LIBERO task adapters")
    parser.add_argument("--catalog", type=Path, default=Path("configs/tasks/catalog.toml"))
    parser.add_argument("--list", action="store_true", help="List tasks without simulation")
    parser.add_argument("--task", default="all",
                        help="all, pick_place, stack, open_drawer, shelf_place")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--init-state-index", type=int, default=0)
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/day04"))
    args = parser.parse_args(argv)
    try:
        catalog = load_task_catalog(args.catalog)
        if args.task != "all" and args.task not in catalog:
            raise ValueError(f"unknown task: {args.task}")
        if not 0 <= args.seed < 2**32 or args.init_state_index < 0:
            raise ValueError("invalid seed or initial state index")
        selected = catalog if args.task == "all" else {args.task: catalog[args.task]}
        if any(not 0 <= args.instruction_index < len(s.instructions) for s in selected.values()):
            raise ValueError("instruction index out of range")
        if args.list:
            print(json.dumps({k: asdict(v) for k, v in selected.items()}, indent=2))
            return 0
        if platform.system() != "Linux":
            raise ValueError("simulation requires Linux/WSL2; --list works on any platform")
        config_template = LiberoSmokeConfig(
            suite="libero_90", task_id=0, seed=args.seed, init_state_index=args.init_state_index,
            max_steps=400, control_frequency=20, settle_steps=10,
            camera_names=("agentview_image", "robot0_eye_in_hand_image"), width=256, height=256,
            video_fps=20, output_dir=args.output_dir,
        )
        if args.output_dir.is_absolute() or ".." in args.output_dir.parts:
            raise ValueError("output directory must be repository-relative without '..'")
        root = resolve_artifact_root(config_template, Path.cwd())
    except (OSError, ValueError) as exc:
        parser.exit(2, f"task configuration error: {exc}\n")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    prepare_libero_config(root)
    from robust_vla_recovery.envs.libero_tasks import LiberoTaskBackend

    records = []
    batch = {"schema_version": 1, "catalog_sha256": sha256_file(args.catalog),
             "seed": args.seed, "runs": records}
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    summary = root / f"summary_{timestamp}.json"
    for key, spec in selected.items():
        adapter = None
        try:
            backend = LiberoTaskBackend(spec)
            adapter = TaskAdapter(spec, backend, instruction_index=args.instruction_index,
                                  init_state_index=args.init_state_index)
            config = LiberoSmokeConfig(
                **{**asdict(config_template), "task_id": backend.task_id,
                   "max_steps": spec.horizon, "control_frequency": spec.control_frequency,
                   "settle_steps": spec.settle_steps, "video_fps": spec.control_frequency}
            )
            print(f"Running {key}: {spec.suite}/{backend.task_id}", flush=True)
            metadata_path = run_libero_smoke(
                config, Path.cwd(), env_factory=lambda _: adapter,
                task_metadata={"key": key, "instruction_index": args.instruction_index,
                               "native_language": backend.native_language,
                               "goal": spec.goal, "bddl_sha256": backend.bddl_sha256,
                               "stability_steps": spec.stability_steps,
                               "catalog_sha256": batch["catalog_sha256"]},
            )
            verified = verify_episode_artifacts(metadata_path)
            records.append({"task": key, "status": "verified", "libero_task_id": backend.task_id,
                            "metadata": str(metadata_path.relative_to(root)),
                            "steps": verified["steps"], "frames": verified["frames"],
                            "success": verified["success"], "final_info": adapter.last_info})
            print(f"Verified {key}: {verified['steps']} steps, {verified['frames']} frames",
                  flush=True)
        except Exception as exc:
            records.append({"task": key, "status": "error", "error": repr(exc),
                            "events": adapter.events if adapter else []})
            print(f"ERROR {key}: {exc}", flush=True)
        finally:
            if adapter is not None:
                adapter.close()
            summary.write_text(json.dumps(batch, indent=2, default=_json_default) + "\n",
                               encoding="utf-8")
    print(f"Summary: {summary}", flush=True)
    return 0 if all(record["status"] == "verified" for record in records) else 1
