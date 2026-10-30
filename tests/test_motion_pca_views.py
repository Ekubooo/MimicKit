"""Local-pose display contracts independent of a GL window."""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

from tools.motion_analysis.player.environment import _environment_types
from tools.motion_analysis.player.controls import PlaybackControls
from tools.motion_analysis.player.view import build_layout, cumulative_selections


class ViewContracts(unittest.TestCase):
    def test_classroom_has_one_upper_and_three_equal_depth_lower_slots(self):
        layout = build_layout("classroom", .9)
        forward = layout.camera_target - layout.camera_position
        forward /= np.linalg.norm(forward)
        right = np.array([.8, .6, 0.])
        up = np.cross(right, forward)
        np.testing.assert_allclose(layout.offsets @ forward, 0, atol=2e-7)
        lower_heights = layout.offsets[1:] @ up
        np.testing.assert_allclose(lower_heights, -3, atol=2e-7)
        np.testing.assert_allclose(layout.offsets[1:] @ right, [-2.5, 0, 2.5], atol=2e-7)
        self.assertEqual(layout.visible, (True, True, True, True))
        self.assertEqual(build_layout("local", .9).visible, (True, True, False, False))

    def test_small_rank_and_constant_data_selections(self):
        self.assertEqual(cumulative_selections(1), ((0,), (0,), (0,)))
        self.assertEqual(cumulative_selections(0), ((), (), ()))
        self.assertEqual(cumulative_selections(5), ((0,), (0, 1), (0, 1, 2)))

    def test_view_requests_preserve_custom_selection_and_do_no_pca_work(self):
        session = SimpleNamespace(variance_ratio=np.array([.6, .3, .1]), max_components=3)
        controls = PlaybackControls(session, (0, 2), "subset")
        controls.component_text = "2-3"
        for view in ("local", "classroom", "comparison"):
            controls.request_view(view)
            self.assertEqual(controls.take_view_request(), view)
            controls.view = view
            self.assertEqual(controls.active_indices, (0, 2))
            self.assertEqual(controls.component_text, "2-3")
            self.assertEqual(controls.mode, "subset")
            self.assertIsNone(controls.take_selection())
        controls.request_camera_reset()
        self.assertTrue(controls.take_camera_reset())
        self.assertFalse(controls.take_camera_reset())

    def test_classroom_hides_selection_and_save_controls(self):
        session = SimpleNamespace(variance_ratio=np.array([.6, .3, .1]), max_components=3,
                                  source=SimpleNamespace(path=Path("test.pkl")), fit_cache_hit=True)
        controls = PlaybackControls(session, (0,), "cumulative", view="classroom")
        buttons, labels = [], []
        ui = SimpleNamespace(
            get_io=lambda: SimpleNamespace(set_ini_filename=lambda path: None),
            separator=lambda: None, same_line=lambda: None,
            text=labels.append, text_wrapped=labels.append,
            button=lambda label: buttons.append(label) or False,
            combo=lambda label, value, options: (False, value),
        )
        controls.draw(ui)
        self.assertNotIn("Save reconstruction", buttons)
        self.assertNotIn("Mean features only", buttons)
        self.assertIn("Restore standard camera", buttons)
        self.assertTrue(any("K=3" in text and "100.00%" in text for text in labels))


class LocalSamplingContracts(unittest.TestCase):
    def test_fixed_root_after_loop_sampling_preserves_joints_and_source(self):
        import torch
        from anim.kin_char_model import Joint, JointType, KinCharModel

        library_type, _ = _environment_types()
        model = KinCharModel("cpu")
        model.init(["root", "hinge"], [-1, 0], [[0, 0, 0], [0, 0, 1]],
                   [[0, 0, 0, 1], [0, 0, 0, 1]],
                   [Joint("root", JointType.ROOT, None),
                    Joint("hinge", JointType.HINGE, torch.tensor([0., 0., 1.]))])
        frames = np.zeros((12, 7))
        frames[:, :3] = np.linspace([2, -1, .9], [6, 2, 1.2], 12)
        frames[:, 5] = np.linspace(.2, 1.5, 12)
        frames[:, 6] = np.linspace(-.5, .5, 12)
        source = SimpleNamespace(path=Path("memory.pkl"), frames=frames.copy(), fps=30., loop_mode=1)
        library = library_type(source, frames.copy(), model, "cpu", (frames.copy(),) * 3)
        ids = torch.tensor([0, 2, 3, 4])
        times = torch.tensor([0., .2, .8, 1.5])
        original = library.calc_motion_frame(ids, times)
        library.local_root_height = .9
        local = library.calc_motion_frame(ids, times)
        torch.testing.assert_close(local[0], torch.tensor([[0., 0., .9]] * 4))
        torch.testing.assert_close(local[1], torch.tensor([[0., 0., 0., 1.]] * 4))
        torch.testing.assert_close(local[4], original[4])
        for velocity in local[2:4]:
            torch.testing.assert_close(velocity, torch.zeros_like(velocity))
        np.testing.assert_array_equal(source.frames, frames)
        library.local_root_height = None
        restored = library.calc_motion_frame(ids, times)
        for before, after in zip(original, restored):
            torch.testing.assert_close(before, after)


if __name__ == "__main__":
    unittest.main()
