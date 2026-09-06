import tempfile
import textwrap
import unittest
from pathlib import Path

from robust_vla_recovery.envs.libero_smoke import (
    LiberoSmokeConfigError,
    describe_value,
    load_libero_smoke_config,
    resolve_artifact_root,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "libero_smoke.toml"


class LiberoSmokeConfigTest(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_libero_smoke_config(DEFAULT_CONFIG)
        self.assertEqual(config.suite, "libero_spatial")
        self.assertEqual(config.task_id, 0)
        self.assertEqual(config.max_steps, 280)
        self.assertEqual(config.camera_names, ("agentview_image", "robot0_eye_in_hand_image"))
        self.assertEqual(resolve_artifact_root(config, REPO_ROOT), REPO_ROOT / "outputs" / "day03")

    def test_unknown_key_is_rejected(self) -> None:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8") + "\nunknown = true\n"
        with self._temporary_config(text) as path:
            with self.assertRaisesRegex(LiberoSmokeConfigError, "unknown key"):
                load_libero_smoke_config(path)

    def test_unsafe_output_path_is_rejected(self) -> None:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            'output_dir = "outputs/day03"', 'output_dir = "../outside"'
        )
        with self._temporary_config(text) as path:
            with self.assertRaisesRegex(LiberoSmokeConfigError, "must not escape"):
                load_libero_smoke_config(path)

    def test_value_description_and_sha256(self) -> None:
        class ArrayLike:
            shape = (2, 3)
            dtype = "float32"

        self.assertEqual(
            describe_value({"state": ArrayLike()}),
            {"state": {"shape": [2, 3], "dtype": "float32"}},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    @staticmethod
    def _temporary_config(text: str):
        class TemporaryConfig:
            def __enter__(self) -> Path:
                self.directory = tempfile.TemporaryDirectory()
                path = Path(self.directory.name) / "config.toml"
                path.write_text(textwrap.dedent(text), encoding="utf-8")
                return path

            def __exit__(self, exc_type, exc_value, traceback) -> None:
                self.directory.cleanup()

        return TemporaryConfig()


if __name__ == "__main__":
    unittest.main()
