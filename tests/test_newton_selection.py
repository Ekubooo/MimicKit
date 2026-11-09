"""Prevent accidental backend/environment mismatches and silent physics overrides."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

from engines.newton_version import check_newton_version


class BackendSelection(unittest.TestCase):
    def test_matching_versions_and_actionable_mismatch(self):
        for backend, version in (("newton", "1.0.0"), ("newton_1_7", "1.7.0")):
            with patch("engines.newton_version.version", return_value=version):
                check_newton_version(backend)
                other = "newton" if backend == "newton_1_7" else "newton_1_7"
                with self.assertRaisesRegex(RuntimeError, "Conda environment"):
                    check_newton_version(other)

    def test_cross_version_evaluation_requires_explicit_selection(self):
        from tools.g1_dodgeball.evaluate import select_engine_config
        from tools.getup.common import read_yaml
        legacy = read_yaml("data/engines/newton_g1_dodgeball_engine.yaml")
        modern = read_yaml("data/engines/newton_1_7_g1_dodgeball_engine.yaml")
        self.assertEqual(select_engine_config(legacy), legacy)
        self.assertEqual(select_engine_config(modern), modern)
        self.assertEqual(select_engine_config(legacy, "newton_1_7"), modern)
        self.assertEqual(select_engine_config(modern, "newton"), legacy)
        with self.assertRaises(ValueError):
            select_engine_config(dict(legacy, sim_freq=120), "newton_1_7")


if __name__ == "__main__":
    unittest.main()
