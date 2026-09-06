"""Opt-in real-simulator checks for reset reproducibility and native success.

The drawer success fixture directly changes a simulator joint. It validates
the predicate/adapter contract, not policy capability, and is never a rollout
used for success-rate measurement.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from robust_vla_recovery.envs.libero_smoke import prepare_libero_config
from robust_vla_recovery.envs.task_adapter import TaskAdapter, load_task_catalog


def main() -> int:
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    root = Path("outputs/day04").resolve()
    prepare_libero_config(root)

    import numpy as np

    from robust_vla_recovery.envs.libero_tasks import LiberoTaskBackend

    records = []
    catalog = load_task_catalog("configs/tasks/catalog.toml")
    for key, spec in catalog.items():
        backend = LiberoTaskBackend(spec)
        adapter = TaskAdapter(spec, backend)
        record = {"task": key, "libero_task_id": backend.task_id}
        try:
            obs1, _ = adapter.reset(seed=20260905)
            state1 = backend.env._env.get_sim_state().copy()
            adapter.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            obs2, _ = adapter.reset(seed=20260905)
            state2 = backend.env._env.get_sim_state().copy()
            delta = float(np.max(np.abs(state1 - state2)))
            pixels_match = all(np.array_equal(obs1["pixels"][camera], obs2["pixels"][camera])
                               for camera in obs1["pixels"])
            record.update(reset_max_abs_delta=delta, reset_images_equal=pixels_match)
            if delta > 1e-9 or not pixels_match:
                raise AssertionError(f"same seed/index reset is not reproducible: {record}")

            # Reject invalid indices before stepping instead of silently taking modulo.
            try:
                backend.reset(seed=20260905, init_state_index=backend.init_state_count)
            except ValueError:
                record["invalid_index_rejected"] = True
            else:
                raise AssertionError("out-of-range initial state was accepted")
            adapter.report_failure("stalled")
            result = adapter.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            if result[2] or result[3]:
                raise AssertionError("recoverable stalled event prematurely ended the episode")
            record["recoverable_event_continues"] = True
            adapter.report_failure("operator_abort")
            try:
                adapter.step(np.zeros(7, dtype=np.float32))
            except RuntimeError:
                record["abort_blocks_step"] = True
            else:
                raise AssertionError("step after abort was accepted")

            if key == "open_drawer":
                adapter.reset(seed=20260905)
                domain = backend.env._env.env
                if domain._check_success():
                    raise AssertionError("drawer unexpectedly starts open")
                site = domain.object_sites_dict[spec.goal[1]]
                for joint in site.joints:
                    address = domain.sim.model.get_joint_qpos_addr(joint)
                    domain.sim.data.qpos[address] = -0.15
                domain.sim.forward()
                if not domain._check_success():
                    raise AssertionError("native drawer predicate did not recognize open fixture")
                for index in range(spec.stability_steps):
                    _, _, terminated, truncated, info = adapter.step(
                        np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                    )
                    if truncated or terminated != (index == spec.stability_steps - 1):
                        raise AssertionError(f"unexpected success termination at {index + 1}")
                if not info["is_success"] or not info["stable_success"]:
                    raise AssertionError("drawer success fixture failed stability diagnosis")
                record["injected_drawer_fixture"] = {
                    "purpose": "contract test only; not policy performance",
                    "qpos": -0.15, "native_success": info["native_success"],
                    "stable_success": info["stable_success"],
                    "first_success_step": info["first_success_step"],
                    "termination_step": info["step"],
                }
            record["status"] = "passed"
        except Exception as exc:
            record.update(status="failed", error=repr(exc))
        finally:
            adapter.close()
            records.append(record)
            print(json.dumps(record), flush=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"adapter_checks_{timestamp}.json"
    path.write_text(json.dumps({"checks": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Checks: {path}", flush=True)
    return 0 if all(record["status"] == "passed" for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
