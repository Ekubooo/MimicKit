"""The saved-analysis CLI only loads snapshots and delegates to the viewer."""

import contextlib
import hashlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.motion_analysis.core.collection import fit_collection, save_collection
from tools.motion_analysis.data.motion import MotionData
from tools.motion_analysis.play_analysis import main


class SavedPlaybackCLIContracts(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(temporary.cleanup)
        self.manifest = Path(temporary.name) / "analysis.json"
        rng = np.random.default_rng(8)
        clips = []
        for i in range(3):
            frames = np.column_stack((np.zeros((10, 6)), rng.normal(size=(10, 3))))
            clips.append(MotionData(Path(f"generated_{i}.pkl"), frames, 30, 0,
                                    hashlib.sha256(frames.tobytes()).hexdigest()))
        analysis = fit_collection(clips, mode="window", window=10)
        save_collection(analysis, self.manifest, group="stage_49", basis="shared")

    def call_main(self, extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(["--manifest", str(self.manifest), "--group", "stage_49",
                         "--mode", "window", *extra])

    def test_load_only_three_views_and_component_modes(self):
        cases = [(["--components", "1", "--view", "comparison"], (0,), "cumulative", "comparison"),
                 (["--selection", "single", "--components", "2", "--view", "local"],
                  (1,), "single", "local"),
                 (["--selection", "subset", "--pcs", "1-2", "--view", "classroom"],
                  (0, 1), "subset", "classroom")]
        for arguments, indices, selection, view in cases:
            with self.subTest(view=view), \
                    patch("tools.motion_analysis.core.collection.fit_pca", side_effect=AssertionError("no fit")), \
                    patch("numpy.linalg.svd", side_effect=AssertionError("no SVD")), \
                    patch("tools.motion_analysis.player.environment.play_session") as play:
                before = Path.cwd()
                self.assertEqual(self.call_main(["--clip", "1", *arguments]), 0)
                self.assertEqual(Path.cwd(), before)
                session, actual_indices = play.call_args.args
                self.assertEqual(actual_indices, indices)
                self.assertEqual(session.source.path.name, "generated_1.pkl")
                self.assertEqual(session.scores.shape[0], 1)
                self.assertEqual(play.call_args.kwargs["selection_mode"], selection)
                self.assertEqual(play.call_args.kwargs["view"], view)
                self.assertTrue(session.fit_cache_hit)

    def test_invalid_clip_and_selection_never_open_viewer(self):
        for arguments in (["--clip", "9"], ["--clip", "-1"],
                          ["--selection", "subset"], ["--pcs", "1"],
                          ["--selection", "subset", "--pcs", "100"],
                          ["--selection", "subset", "--pcs", "1", "--components", "1"]):
            with self.subTest(arguments=arguments), \
                    patch("tools.motion_analysis.player.environment.play_session") as play:
                self.assertEqual(self.call_main(arguments), 1)
                play.assert_not_called()

    def test_playback_failure_restores_working_directory_and_preserves_snapshot(self):
        before = Path.cwd()
        files = {path: path.read_bytes() for path in self.manifest.parent.rglob("*") if path.is_file()}
        with patch("tools.motion_analysis.player.environment.play_session", side_effect=RuntimeError("GL unavailable")):
            self.assertEqual(self.call_main([]), 1)
        self.assertEqual(Path.cwd(), before)
        self.assertEqual(files, {path: path.read_bytes() for path in files})


if __name__ == "__main__":
    unittest.main()
