"""CLI contracts independent of the Newton viewer and real motion assets."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.motion_analysis import run_pca


class PcaCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.motion_root = self.root / "motions"
        source = self.motion_root / "humanoid" / "sample.pkl"
        source.parent.mkdir(parents=True)
        rng = np.random.default_rng(24)
        frames = rng.normal(scale=0.1, size=(18, 10))
        with source.open("wb") as stream:
            pickle.dump({"frames": frames, "fps": 60, "loop_mode": 0}, stream)
        self.source = source
        self.output = io.StringIO()
        self.errors = io.StringIO()

    def invoke(self, *arguments):
        with (
            patch.object(run_pca, "MOTION_ROOT", self.motion_root),
            patch.object(run_pca, "CACHE_ROOT", self.root / "cache"),
            patch("tools.motion_analysis.reporting.reconstruction.generate_report", return_value={}),
            redirect_stdout(self.output),
            redirect_stderr(self.errors),
        ):
            return run_pca.main(["--motion", "humanoid/sample.pkl", *arguments])

    def test_cli_reuses_fit_for_different_component_selections(self):
        previous_newton = {name for name in sys.modules if name == "newton" or name.startswith("newton.")}
        self.assertEqual(self.invoke("--components", "2"), 0, self.errors.getvalue())
        self.assertIn("Fit cache: computed", self.output.getvalue())
        self.output.truncate(0)
        self.output.seek(0)
        self.assertEqual(self.invoke("--selection", "subset", "--pcs", "1,3"), 0, self.errors.getvalue())
        self.assertIn("Fit cache: hit", self.output.getvalue())
        self.assertIn("Selected components: 1,3", self.output.getvalue())
        self.assertEqual(
            previous_newton,
            {name for name in sys.modules if name == "newton" or name.startswith("newton.")},
        )

    def test_explicit_invalid_component_is_not_silently_clamped(self):
        self.assertEqual(self.invoke("--components", "99"), 1)
        self.assertIn("PCA error:", self.errors.getvalue())

    def test_nonfinite_input_reports_locations_before_fitting_or_playback(self):
        data = pickle.loads(self.source.read_bytes())
        data["frames"][2, 7] = np.nan
        data["frames"][5, 8] = np.inf
        original = pickle.dumps(data)
        self.source.write_bytes(original)
        with patch("tools.motion_analysis.core.session.fit_pca", side_effect=AssertionError("Fitted invalid input")):
            self.assertEqual(self.invoke("--mode", "window", "--play"), 1)
        message = self.errors.getvalue()
        self.assertIn(str(self.source), message)
        self.assertIn("NaN=1, Inf=1", message)
        self.assertIn("affected frames=2/18", message)
        self.assertIn("columns (0-based)=[7, 8]", message)
        self.assertIn("[[2, 7], [5, 8]]", message)
        self.assertFalse((self.root / "cache").exists())
        self.assertEqual(self.source.read_bytes(), original)

    def test_subset_requires_its_own_selection_argument(self):
        self.assertEqual(self.invoke("--selection", "subset"), 1)
        self.assertIn("requires --pcs", self.errors.getvalue())
        self.assertEqual(self.invoke("--selection", "subset", "--pcs", "1", "--components", "2"), 1)

    def test_empty_subset_is_mean_only(self):
        self.assertEqual(self.invoke("--selection", "subset", "--pcs", ""), 0, self.errors.getvalue())
        self.assertIn("Selected components: mean only", self.output.getvalue())

    def test_playback_view_is_forwarded_without_affecting_offline_selection(self):
        with patch("tools.motion_analysis.player.environment.play_session") as play:
            self.assertEqual(self.invoke("--view", "classroom", "--play", "--components", "2"), 0)
        self.assertEqual(play.call_args.kwargs["view"], "classroom")
        self.assertEqual(play.call_args.args[1], (0, 1))

    def test_input_resolution_is_independent_of_working_directory(self):
        with patch.object(run_pca, "MOTION_ROOT", self.motion_root):
            self.assertEqual(run_pca.resolve_motion_path("humanoid/sample.pkl"), self.source)
            for invalid in ("../sample.pkl", str(self.source), "missing.pkl"):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    run_pca.resolve_motion_path(invalid)


if __name__ == "__main__":
    unittest.main()
