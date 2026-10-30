"""Opt-in saved-case playback acceptance, one native GL process per group."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))


@unittest.skipUnless(os.environ.get("MIMICKIT_SMP_CASE_GL") == "1", "Opt-in SMP case GL test")
class SavedCaseGL(unittest.TestCase):
    def test_saved_only_cli_and_native_controls(self):
        import torch
        from envs.env_builder import load_configs
        from envs.sim_env import PlayMode
        from tools.motion_analysis.play_analysis import main
        from tools.motion_analysis.player.environment import _environment_types
        from tools.motion_analysis.player.controls import PlaybackControls

        group = os.environ.get("MIMICKIT_SMP_CASE_GROUP", "reference")
        mode = os.environ.get("MIMICKIT_SMP_CASE_MODE", "pose")
        index = ROOT / "report/humanoid_spinkick/assets/data/analysis.json"
        before = index.read_bytes()
        visited = []

        def bounded_play(session, indices, *, selection_mode, env_config, engine_config,
                         device, view):
            environment, engine = load_configs(str(env_config), str(engine_config))
            controls = PlaybackControls(session, indices, selection_mode, view=view)
            _, environment_type = _environment_types()
            env = environment_type.__new__(environment_type)
            try:
                env.__init__(session, controls, environment, engine, device)
                env.reset()
                action = torch.zeros((4, *env.get_action_space().shape), device=device)
                with torch.no_grad():
                    env.step(action)
                controls.request_action("pause")
                env._render_scene()
                self.assertEqual(env._play_mode, PlayMode.PAUSE)
                paused = env._time_buf.clone()
                model = env._engine.get_sim_model()
                for target in ("local", "comparison", "classroom", "local"):
                    controls.request_view(target)
                    env._render_scene()
                    torch.testing.assert_close(env._time_buf, paused)
                    self.assertIs(env._engine.get_sim_model(), model)
                    self.assertTrue(torch.isfinite(env._engine.get_body_pos(env._get_char_id())).all())
                controls.mode, controls.k = "cumulative", min(2, session.max_components)
                self.assertTrue(controls.request_selection())
                env._render_scene()
                torch.testing.assert_close(env._time_buf, paused)
                controls.request_action("step")
                env._render_scene()
                with torch.no_grad():
                    env.step(action)
                torch.testing.assert_close(env._time_buf - paused,
                                           torch.full_like(paused, env._engine.get_timestep()))
                controls.request_action("restart")
                env._render_scene()
                np.testing.assert_array_equal(env._time_buf.cpu().numpy(), np.zeros(4))
                visited.append(session.source_id)
            finally:
                env.close()
            self.assertIsNone(env._engine.get_viewer())

        with patch("numpy.linalg.svd", side_effect=AssertionError("Playback attempted PCA fitting")), \
                patch("tools.motion_analysis.player.environment.play_session", side_effect=bounded_play):
            result = main(["--manifest", str(index), "--group", group, "--mode", mode,
                           "--view", "classroom", "--device", "cuda:0"])
        self.assertEqual(result, 0)
        self.assertEqual(len(visited), 1)
        self.assertEqual(index.read_bytes(), before)
        self.assertFalse(any(name.startswith("tools.motion_analysis.smp.sampling") for name in sys.modules))
        print(f"Saved-case GL passed: {group}/{mode}, no SVD or sampling imports")


if __name__ == "__main__":
    unittest.main()
