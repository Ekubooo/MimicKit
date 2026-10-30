"""Offline resampling uses the source clock without changing source assets."""

import hashlib
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.motion_analysis.data.motion import read_motion
from tools.motion_analysis.data.resampling import resample_motion


class ResamplingContracts(unittest.TestCase):
    def test_reference_regular_clock_and_source_unchanged(self):
        path = ROOT / "data/motions/humanoid/humanoid_spinkick.pkl"
        character = ROOT / "data/assets/humanoid/humanoid.xml"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        original = read_motion(path)
        result = resample_motion(path, character, fps=30)
        self.assertEqual(result.frames.shape, (39, 34))
        self.assertEqual(result.fps, 30)
        self.assertEqual(result.loop_mode, original.loop_mode)
        # 60 -> 30 Hz samples coincide with every other original sample.
        np.testing.assert_allclose(result.frames[:, :3], original.frames[::2, :3], atol=2e-6)
        self.assertLessEqual((len(result.frames) - 1) / result.fps,
                             (len(original.frames) - 1) / original.fps)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
        np.testing.assert_array_equal(read_motion(path).frames, original.frames)

    def test_unchanged_clock_is_lossless_and_bad_fps_is_rejected(self):
        path = ROOT / "data/motions/humanoid/humanoid_spinkick.pkl"
        character = ROOT / "data/assets/humanoid/humanoid.xml"
        source = read_motion(path)
        same = resample_motion(path, character, fps=source.fps)
        np.testing.assert_array_equal(same.frames, source.frames)
        self.assertIsNot(same.frames, source.frames)
        for fps in (0, -1, np.nan, np.inf):
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                resample_motion(path, character, fps=fps)


if __name__ == "__main__":
    unittest.main()
