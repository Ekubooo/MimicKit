"""PCA selection and playback synchronization contracts without an OpenGL window."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

from tools.motion_analysis.playback import PauseBridge, _environment_types
from tools.motion_analysis.playback_controls import PlaybackControls


class FakeSession:
    def __init__(self):
        self.source = SimpleNamespace(
            path=Path("source.pkl"), frames=np.zeros((12, 7), dtype=np.float32),
            fps=30.0, loop_mode=0)
        self.max_components = 8
        self.variance_ratio = np.array([0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
        self.fit_cache_hit = True
        self.cache_dir = Path("cache")
        self.reconstruction_calls = []
        self.save_calls = []

    def reconstruct(self, indices):
        self.reconstruction_calls.append(tuple(indices))
        frames = self.source.frames.copy()
        frames[:, 6] = sum(index + 1 for index in indices)
        return frames

    def save_reconstruction(self, indices):
        self.save_calls.append(tuple(indices))
        return self.cache_dir / "reconstructed.pkl"


class PlaybackControlTests(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        self.controls = PlaybackControls(self.session, (0, 1), "cumulative")

    def test_three_selection_modes_and_mean(self):
        self.controls.k = 3
        self.assertTrue(self.controls.request_selection())
        self.assertEqual(self.controls.take_selection(), (0, 1, 2))
        self.controls.mode = "single"
        self.assertTrue(self.controls.request_selection())
        self.assertEqual(self.controls.take_selection(), (2,))
        self.controls.mode = "subset"
        self.controls.component_text = "1,3,5-8,3"
        self.assertTrue(self.controls.request_selection())
        self.assertEqual(self.controls.take_selection(), (0, 2, 4, 5, 6, 7))
        self.controls.component_text = ""
        self.assertTrue(self.controls.request_selection())
        self.assertEqual(self.controls.take_selection(), ())
        self.controls.mode, self.controls.k = "cumulative", 0
        self.assertTrue(self.controls.request_selection())
        self.assertEqual(self.controls.take_selection(), ())

    def test_invalid_edit_does_not_replace_active_selection(self):
        self.controls.mode, self.controls.component_text = "subset", "1,900"
        self.assertFalse(self.controls.request_selection())
        self.assertTrue(self.controls.error)
        self.assertEqual(self.controls.active_indices, (0, 1))
        self.assertIsNone(self.controls.take_selection())

    def test_entering_subset_uses_current_selection_without_overwriting_edits(self):
        self.controls.selection_applied(tuple(range(8)))
        self.controls.set_mode("subset")
        self.assertEqual(self.controls.component_text, "1,2,3,4,5,6,7,8")
        self.assertIsNone(self.controls.take_selection())
        self.controls.component_text = "2,4-6"
        self.controls.set_mode("subset")
        self.assertEqual(self.controls.component_text, "2,4-6")

        self.controls.k = 3
        self.controls.set_mode("single")
        indices = self.controls.take_selection()
        self.assertEqual(indices, (2,))
        self.controls.selection_applied(indices)
        self.controls.set_mode("subset")
        self.assertEqual(self.controls.component_text, "3")

    def test_selection_requests_do_not_fit_reconstruct_or_save(self):
        self.controls.k = 4
        self.controls.request_selection()
        self.assertEqual(self.session.reconstruction_calls, [])
        self.assertEqual(self.session.save_calls, [])
        self.controls.selection_applied((0, 1, 2, 3))
        self.assertAlmostEqual(self.controls.selected_variance, 0.8)
        self.controls.request_save()
        self.assertTrue(self.controls.take_save_request())
        self.assertFalse(self.controls.take_save_request())

    def test_keyboard_capture_and_action_queue(self):
        self.assertFalse(self.controls.captures_keyboard)
        self.controls._imgui = SimpleNamespace(
            get_io=lambda: SimpleNamespace(want_capture_keyboard=True))
        self.assertTrue(self.controls.captures_keyboard)
        self.controls.request_action("pause")
        self.controls.request_action("step")
        self.assertEqual(self.controls.take_actions(), ("pause", "step"))
        self.assertEqual(self.controls.take_actions(), ())

    def test_pca_ui_disables_implicit_working_directory_ini(self):
        ini_targets = []
        io = SimpleNamespace(want_capture_keyboard=False,
                             set_ini_filename=ini_targets.append)
        ui = SimpleNamespace(
            get_io=lambda: io,
            get_style=lambda: SimpleNamespace(font_size_base=13.0),
            get_font_size=lambda: 13.0,
            push_font=lambda font, size: None,
            pop_font=lambda: None,
            separator=lambda: None,
            text=lambda text: None,
            text_wrapped=lambda text: None,
            button=lambda label: False,
            same_line=lambda: None,
            combo=lambda label, index, labels: (False, index),
            slider_int=lambda label, value, minimum, maximum: (False, value),
        )
        self.controls.draw(ui)
        self.controls.draw(ui)
        self.assertEqual(ini_targets, [None])

    def test_font_increase_is_local_and_restored_even_on_draw_failure(self):
        # A 2x DPI scale must still add exactly two displayed pixels. Simulate
        # another viewer section after drawing, including the exception path.
        for fail in (False, True):
            with self.subTest(draw_failure=fail):
                style = SimpleNamespace(font_size_base=13.0)
                stack = []

                def push_font(font, size):
                    self.assertIsNone(font)
                    stack.append(style.font_size_base)
                    style.font_size_base = size

                def pop_font():
                    style.font_size_base = stack.pop()

                ui = SimpleNamespace(
                    get_io=lambda: SimpleNamespace(set_ini_filename=lambda path: None),
                    get_style=lambda: style,
                    get_font_size=lambda: style.font_size_base * 2.0,
                    push_font=push_font,
                    pop_font=pop_font,
                )

                def draw_contents(imgui):
                    self.assertAlmostEqual(imgui.get_font_size(), 28.0)
                    if fail:
                        raise RuntimeError("draw failed")

                self.controls._draw_contents = draw_contents
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "draw failed"):
                        self.controls.draw(ui)
                else:
                    self.controls.draw(ui)
                self.assertEqual(stack, [])
                self.assertEqual(ui.get_font_size(), 26.0)


class PauseBridgeTests(unittest.TestCase):
    def test_reads_native_pause_and_publishes_tool_pause(self):
        engine = SimpleNamespace(paused=False)
        engine.set_viewer_paused = lambda value: setattr(engine, "paused", bool(value))
        engine.get_viewer_paused = lambda: engine.paused
        bridge = PauseBridge(engine)
        bridge.publish(True)
        self.assertTrue(bridge.read())
        engine.paused = False  # Native Newton checkbox changed during render.
        self.assertFalse(bridge.read())


class EnvironmentAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory_library, cls.environment_type = _environment_types()

    def make_environment_shell(self):
        import torch
        from envs.sim_env import PlayMode

        env = self.environment_type.__new__(self.environment_type)
        env.session = FakeSession()
        env.controls = PlaybackControls(env.session, (0,), "cumulative")
        env._play_mode = PlayMode.PAUSE
        env._time_buf = torch.tensor([0.2, 0.2])
        engine = SimpleNamespace(paused=True)
        engine.set_viewer_paused = lambda value: setattr(engine, "paused", bool(value))
        engine.get_viewer_paused = lambda: engine.paused
        env._engine = engine
        env._pause_bridge = PauseBridge(engine)
        return env

    def test_step_request_restores_native_pause_after_space_toggle(self):
        from envs.sim_env import PlayMode

        env = self.make_environment_shell()
        env._engine.paused = False  # Newton's SPACE callback fires first.
        env._apply_actions(("step",))
        self.assertEqual(env._play_mode, PlayMode.ONE_STEP)
        self.assertTrue(env._engine.paused)

    def test_pause_and_play_update_both_states_without_advancing_time(self):
        from envs.sim_env import PlayMode

        env = self.make_environment_shell()
        before = env._time_buf.clone()
        env._apply_actions(("play",))
        self.assertEqual(env._play_mode, PlayMode.PLAY)
        self.assertFalse(env._engine.paused)
        env._apply_actions(("pause",))
        self.assertEqual(env._play_mode, PlayMode.PAUSE)
        self.assertTrue(env._engine.paused)
        self.assertTrue((env._time_buf == before).all())

    def test_gui_shortcuts_ignore_text_input(self):
        env = self.make_environment_shell()
        callbacks = {}
        env._engine.register_keyboard_callback = callbacks.__setitem__
        env._engine.register_ui_callback = lambda callback, position: None
        env._setup_gui()
        env.controls._imgui = SimpleNamespace(
            get_io=lambda: SimpleNamespace(want_capture_keyboard=True))
        callbacks["ENTER"]()
        callbacks["SPACE"]()
        self.assertEqual(env.controls.take_actions(), ())
        env.controls._imgui = None
        callbacks["SPACE"]()
        self.assertEqual(env.controls.take_actions(), ("step",))

    def test_invalid_reconstruction_keeps_current_motion_library(self):
        env = self.make_environment_shell()
        previous_library = object()
        env._motion_lib = previous_library
        env.session.reconstruct = lambda indices: np.zeros((2, 99))
        env.controls.k = 2
        env.controls.request_selection()
        env._apply_selection()
        self.assertIs(env._motion_lib, previous_library)
        self.assertEqual(env.controls.active_indices, (0,))
        self.assertTrue(env.controls.error)

    def test_apply_refreshes_paused_pose_at_same_time_without_disk_writes(self):
        import torch
        from anim.kin_char_model import Joint, JointType, KinCharModel
        from envs.sim_env import PlayMode

        env = self.make_environment_shell()
        model = KinCharModel("cpu")
        root = Joint("root", JointType.ROOT, None)
        hinge = Joint("hinge", JointType.HINGE, torch.tensor([0.0, 0.0, 1.0]))
        model.init(["root", "hinge"], [-1, 0], [[0, 0, 0], [0, 0, 1]],
                   [[0, 0, 0, 1], [0, 0, 0, 1]], [root, hinge])
        env._kin_char_model, env._device = model, "cpu"
        refreshed = []
        env._sync_motion = lambda: refreshed.append(env._time_buf.clone())
        before = env._time_buf.clone()
        env.controls.k = 2
        env.controls.request_selection()
        env._apply_selection()
        self.assertEqual(env.controls.active_indices, (0, 1))
        self.assertEqual(env._play_mode, PlayMode.PAUSE)
        self.assertEqual(len(refreshed), 1)
        torch.testing.assert_close(refreshed[0], before)
        torch.testing.assert_close(env._time_buf, before)
        self.assertEqual(env.session.save_calls, [])

    def test_explicit_save_only(self):
        env = self.make_environment_shell()
        env._save_selection()
        self.assertEqual(env.session.save_calls, [])
        env.controls.request_save()
        env._save_selection()
        self.assertEqual(env.session.save_calls, [(0,)])

    def test_in_memory_library_uses_standard_sampling_for_both_characters(self):
        import torch
        from anim.kin_char_model import Joint, JointType, KinCharModel

        model = KinCharModel("cpu")
        root = Joint("root", JointType.ROOT, None)
        hinge = Joint("hinge", JointType.HINGE, torch.tensor([0.0, 0.0, 1.0]))
        model.init(["root", "hinge"], [-1, 0], [[0, 0, 0], [0, 0, 1]],
                   [[0, 0, 0, 1], [0, 0, 0, 1]], [root, hinge])
        source = FakeSession().source
        source.frames[:, 2] = 1.0
        source.frames[:, 6] = np.linspace(0, 1, len(source.frames))
        reconstruction = source.frames.copy()
        reconstruction[:, 6] *= 0.5
        library = self.memory_library(source, reconstruction, model, "cpu")
        ids, times = torch.tensor([0, 1]), torch.tensor([0.15, 0.15])
        pos, rot, _, _, joints, _ = library.calc_motion_frame(ids, times)
        torch.testing.assert_close(pos[0], pos[1])
        torch.testing.assert_close(rot[0], rot[1])
        dof = library.joint_rot_to_dof(joints)
        self.assertAlmostEqual(float(dof[1, 0]), float(dof[0, 0]) / 2, places=5)


if __name__ == "__main__":
    unittest.main()
