import importlib
import unittest


class ImportSmokeTest(unittest.TestCase):
    def test_public_packages_import_without_heavy_dependencies(self) -> None:
        modules = (
            "robust_vla_recovery",
            "robust_vla_recovery.config",
            "robust_vla_recovery.data",
            "robust_vla_recovery.data.rollout",
            "robust_vla_recovery.data.collection",
            "robust_vla_recovery.envs",
            "robust_vla_recovery.envs.task_adapter",
            "robust_vla_recovery.envs.libero_tasks",
            "robust_vla_recovery.envs.task_cli",
            "robust_vla_recovery.policy",
            "robust_vla_recovery.evaluation",
        )
        for module in modules:
            with self.subTest(module=module):
                self.assertIsNotNone(importlib.import_module(module))


if __name__ == "__main__":
    unittest.main()
