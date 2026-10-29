"""Opt-in integration check with a real Newton/OpenGL window and CUDA backend.

Set MIMICKIT_PCA_GL_TEST=1 to run. MIMICKIT_PCA_GL_CASE selects faceup (pose)
or facedown (window). Each case runs in a fresh process because Newton's window
lifecycle is process-scoped. Desktop mouse/keyboard acceptance remains separate.
"""

import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))


@unittest.skipUnless(os.environ.get("MIMICKIT_PCA_GL_TEST") == "1", "Opt-in real GL window test")
class NewtonPcaIntegration(unittest.TestCase):
    def test_selections_pause_step_restart_and_close_in_real_viewer(self):
        import torch
        from envs.env_builder import load_configs
        from envs.sim_env import PlayMode
        from tools.motion_analysis.playback import _environment_types
        from tools.motion_analysis.playback_controls import PlaybackControls
        from tools.motion_analysis.session import load_session

        original_cwd = Path.cwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, original_cwd)
        cache = ROOT / "tools/motion_analysis/PCA_Result"
        environment, engine = load_configs(
            str(ROOT / "data/envs/view_motion_humanoid_env.yaml"),
            str(ROOT / "data/engines/newton_engine.yaml"),
        )
        _, environment_type = _environment_types()
        cases = {"faceup": ("humanoid_getup_faceup.pkl", "pose"),
                 "facedown": ("humanoid_getup_facedown.pkl", "window")}
        selected_case = os.environ.get("MIMICKIT_PCA_GL_CASE", "faceup")
        self.assertIn(selected_case, cases)
        for filename, mode in (cases[selected_case],):
            with self.subTest(motion=filename, mode=mode):
                session = load_session(ROOT / "data/motions/humanoid" / filename,
                                       mode=mode, cache_root=cache)
                controls = PlaybackControls(session, (0, 1, 2, 3), "cumulative")
                env = environment_type.__new__(environment_type)
                try:
                    env.__init__(session, controls, dict(environment), dict(engine), "cuda:0")
                    env.reset()
                    action = torch.zeros((2, *env.get_action_space().shape), device="cuda:0")
                    with torch.no_grad():
                        env.step(action)
                    self.assertEqual(env._time_buf[0].item(), env._time_buf[1].item())

                    controls.request_action("pause")
                    env._render_scene()
                    self.assertEqual(env._play_mode, PlayMode.PAUSE)
                    self.assertTrue(env._engine.get_viewer_paused())
                    paused_time = env._time_buf.clone()
                    original_body = env._engine.get_body_pos(env._get_char_id())[0].clone()

                    # Cover cumulative, isolated component and arbitrary subset.
                    for selection_mode, k, text, expected in (
                        ("cumulative", 2, "", (0, 1)),
                        ("single", 3, "", (2,)),
                        ("subset", 3, "1,3,5-8", (0, 2, 4, 5, 6, 7)),
                    ):
                        controls.mode, controls.k, controls.component_text = selection_mode, k, text
                        self.assertTrue(controls.request_selection())
                        env._render_scene()
                        env._render_scene()
                        self.assertFalse(controls.error)
                        self.assertEqual(controls.active_indices, expected)
                        torch.testing.assert_close(env._time_buf, paused_time)
                        body = env._engine.get_body_pos(env._get_char_id())
                        torch.testing.assert_close(body[0], original_body)
                        self.assertTrue(bool(torch.isfinite(body).all()))
                        roots = env._engine.get_root_pos(env._get_char_id())
                        torch.testing.assert_close(roots[0], roots[1])

                    controls.request_action("step")
                    env._render_scene()
                    with torch.no_grad():
                        env.step(action)
                    self.assertEqual(env._play_mode, PlayMode.PAUSE)
                    torch.testing.assert_close(
                        env._time_buf - paused_time,
                        torch.full_like(paused_time, env._engine.get_timestep()),
                    )
                    controls.request_action("restart")
                    env._render_scene()
                    np.testing.assert_array_equal(env._time_buf.cpu().numpy(), [0, 0])

                    # Native controls are drawn inside end_frame; emulate that
                    # event at the same application callback boundary.
                    native_request = []

                    def native_pause_callback(imgui):
                        if native_request:
                            env._engine.set_viewer_paused(native_request.pop())

                    env._engine.register_ui_callback(native_pause_callback, position="free")
                    controls.request_action("play")
                    env._render_scene()
                    native_request.append(True)
                    env._render_scene()
                    self.assertEqual(env._play_mode, PlayMode.PAUSE)
                    self.assertTrue(controls.paused)
                    print(f"Real GL verified: {filename}, {mode}, three selection modes, pause/step/restart")
                finally:
                    env.close()
                self.assertIsNone(env._engine._viewer)


if __name__ == "__main__":
    unittest.main()
