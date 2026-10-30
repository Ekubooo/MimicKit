"""Opt-in integration check with a real Newton/OpenGL window and CUDA backend.

Set MIMICKIT_PCA_GL_TEST=1 to run. MIMICKIT_PCA_GL_CASE selects faceup, facedown,
walk or walk_window. Each case runs in a fresh process because Newton's window
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
                 "facedown": ("humanoid_getup_facedown.pkl", "window"),
                 "walk": ("long/humanoid_long_walk0.pkl", "pose"),
                 "walk_window": ("long/humanoid_long_walk0.pkl", "window")}
        selected_case = os.environ.get("MIMICKIT_PCA_GL_CASE", "faceup")
        self.assertIn(selected_case, cases)
        for filename, mode in (cases[selected_case],):
            with self.subTest(motion=filename, mode=mode):
                session = load_session(ROOT / "data/motions/humanoid" / filename,
                                       mode=mode, cache_root=cache)
                controls = PlaybackControls(session, (0, 1, 2, 3), "cumulative",
                                            view=os.environ.get("MIMICKIT_PCA_GL_VIEW", "comparison"))
                pca_font_sizes = []
                draw_contents = controls._draw_contents

                def check_pca_font(imgui):
                    pca_font_sizes.append(imgui.get_font_size())
                    draw_contents(imgui)

                controls._draw_contents = check_pca_font
                env = environment_type.__new__(environment_type)
                try:
                    env.__init__(session, controls, dict(environment), dict(engine), "cuda:0")
                    env.reset()
                    action = torch.zeros((env.get_num_envs(), *env.get_action_space().shape), device="cuda:0")
                    with torch.no_grad():
                        env.step(action)
                    self.assertTrue(pca_font_sizes)
                    np.testing.assert_allclose(pca_font_sizes, 18.0)
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

                    from tools.motion_analysis.playback_view import build_layout

                    model = env._engine.get_sim_model()
                    model_scales = model.shape_scale.numpy().copy()
                    library = env._motion_lib
                    active = controls.active_indices
                    preview_files = set(session.cache_dir.iterdir())
                    for view in ("local", "classroom", "comparison", "classroom", "local", "comparison"):
                        controls.request_view(view)
                        env._render_scene()
                        env._render_scene()
                        torch.testing.assert_close(env._time_buf, paused_time)
                        self.assertIs(env._engine.get_sim_model(), model)
                        self.assertIs(env._motion_lib, library)
                        np.testing.assert_array_equal(model.shape_scale.numpy(), model_scales)
                        self.assertEqual(controls.active_indices, active)
                        expected = build_layout(view, float(session.source.frames[0, 2]))
                        np.testing.assert_allclose(env._engine.get_viewer().world_offsets.numpy(), expected.offsets)
                        self.assertEqual(env._display_renderer.visible, expected.visible)
                        for batch, masked in env._display_renderer._masked_batches:
                            worlds = batch.worlds.numpy()
                            hidden = np.array([not expected.visible[w] if w >= 0 else view == "classroom"
                                               for w in worlds])
                            np.testing.assert_array_equal(masked.numpy()[hidden], 0)
                        if view != "comparison":
                            roots = env._engine.get_root_pos(env._get_char_id())
                            target = torch.tensor([0., 0., session.source.frames[0, 2]],
                                                  dtype=roots.dtype, device=roots.device)
                            torch.testing.assert_close(roots, target.expand_as(roots))
                            viewer = env._engine.get_viewer()
                            viewer.camera.pitch += 20  # Emulate mouse input before the draw hook.
                            env._render_scene()
                            np.testing.assert_allclose(env._engine.get_camera_pos(), env._camera.pos, atol=1e-5)
                            controls.request_camera_reset()
                            env._render_scene()
                            self.assertTrue(all(line.hidden for line in env._engine.get_viewer().lines.values()))
                        if view == "classroom":
                            self.assertEqual(env._get_env_motion_ids().tolist(), [0, 2, 3, 4])
                        if os.environ.get("MIMICKIT_PCA_GL_CAPTURE") == "1":
                            from PIL import Image

                            env._render_scene()
                            image = env._engine.get_viewer().get_frame(render_ui=True).numpy()
                            Image.fromarray(image).save(ROOT / f"output/pca_{view}_{selected_case}.png")
                    self.assertEqual(set(session.cache_dir.iterdir()), preview_files)

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
                    np.testing.assert_array_equal(env._time_buf.cpu().numpy(), [0, 0, 0, 0])

                    # Joint motion advances in classroom mode while every root
                    # and all four clocks stay fixed/synchronized as specified.
                    controls.request_view("classroom")
                    env._render_scene()
                    before_joints = env._engine.get_dof_pos(env._get_char_id()).clone()
                    for _ in range(5):
                        controls.request_action("step")
                        env._render_scene()
                        with torch.no_grad():
                            env.step(action)
                    roots = env._engine.get_root_pos(env._get_char_id())
                    torch.testing.assert_close(roots[:, :2], torch.zeros_like(roots[:, :2]))
                    torch.testing.assert_close(roots[:, 2], torch.full_like(roots[:, 2], session.source.frames[0, 2]))
                    torch.testing.assert_close(env._time_buf, env._time_buf[0].expand_as(env._time_buf))
                    self.assertFalse(torch.allclose(before_joints, env._engine.get_dof_pos(env._get_char_id())))
                    controls.request_view("comparison")
                    env._render_scene()

                    # Native controls are drawn inside end_frame; emulate that
                    # event at the same application callback boundary.
                    native_request = []

                    def native_pause_callback(imgui):
                        self.assertAlmostEqual(imgui.get_font_size(), 18.0)
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
