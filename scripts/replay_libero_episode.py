"""Verify and decode a saved LIBERO episode without a display server."""

from robust_vla_recovery.envs.libero_smoke import replay_main


if __name__ == "__main__":
    raise SystemExit(replay_main())
