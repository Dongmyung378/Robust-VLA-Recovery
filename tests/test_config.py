import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from robust_vla_recovery.config import ConfigError, load_config, resolve_output_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.toml"


class ConfigTest(unittest.TestCase):
    def test_default_config_is_valid(self) -> None:
        config = load_config(DEFAULT_CONFIG)
        self.assertEqual(config.run.seed, 378)
        self.assertEqual(config.run.device, "auto")
        self.assertEqual(len(config.project.tasks), 4)
        self.assertEqual(resolve_output_dir(config, REPO_ROOT), REPO_ROOT / "outputs")

    def test_unknown_key_is_rejected(self) -> None:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8") + "\nunknown = true\n"
        with self._temporary_config(text) as path:
            with self.assertRaisesRegex(ConfigError, "unknown key"):
                load_config(path)

    def test_parent_output_path_is_rejected(self) -> None:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            'output_dir = "outputs"', 'output_dir = "../outside"'
        )
        with self._temporary_config(text) as path:
            with self.assertRaisesRegex(ConfigError, "must not escape"):
                load_config(path)

    def test_duplicate_task_is_rejected(self) -> None:
        text = DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            '"shelf_place"]', '"pick_place"]'
        )
        with self._temporary_config(text) as path:
            with self.assertRaisesRegex(ConfigError, "duplicates"):
                load_config(path)

    @staticmethod
    @contextmanager
    def _temporary_config(text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            path.write_text(text, encoding="utf-8")
            yield path


if __name__ == "__main__":
    unittest.main()
