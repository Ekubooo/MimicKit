"""Feature/diagnostic contracts; no simulator or trained-policy success assertions."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from tools.diffusion_model.check_getup_prior import (
    FIELDS, MotionPriorData, build_cases, observe, perturb_joints,
    score_windows, transform_world,
)


ASSETS = [ROOT / "data/assets/humanoid/humanoid.xml",
          ROOT / "data/motions/humanoid/humanoid_getup_faceup.pkl",
          ROOT / "data/motions/humanoid/humanoid_getup_facedown.pkl"]


@unittest.skipUnless(all(path.is_file() for path in ASSETS), "Download Humanoid assets first.")
class GetupDiagnosticContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cfg = yaml.safe_load((ROOT / "tools/diffusion_model/config/tinymdm_getup.yaml").read_text())
        cls.dataset = MotionPriorData(cfg, "cpu")
        cls.env = yaml.safe_load((ROOT / cfg["env_config"]).read_text())
        cls.pose_env = yaml.safe_load((ROOT / "data/envs/smp_humanoid_env.yaml").read_text())

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)
        self.ids = torch.tensor([0, 0, 1, 1])
        self.times = torch.tensor([0.0, 1.1, 0.9, 2.9])
        self.reference = dict(zip(
            FIELDS, self.dataset._fetch_smp_demo_data(self.ids, self.times)))

    def test_matches_training_features_including_start_padding(self):
        expected = self.dataset._compute_smp_obs_demo(self.ids, self.times)
        actual = observe(self.dataset, self.reference)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (4, 1140))

    def test_world_yaw_translation_preserves_local_heading_features(self):
        original = observe(self.dataset, self.reference)
        moved = observe(self.dataset, transform_world(self.dataset, self.reference))
        torch.testing.assert_close(moved, original, rtol=1e-4, atol=2e-4)

    def test_static_candidates_and_reference_coverage(self):
        cases, ids = build_cases(self.dataset, 8, self.pose_env, self.env["char_file"])
        self.assertEqual(ids.bincount().tolist(), [8, 8])
        for name in ("standing_init_pose", "static_end_faceup", "static_end_facedown"):
            window = cases[name]
            self.assertEqual(window["root_vel"].count_nonzero().item(), 0)
            self.assertEqual(window["root_ang_vel"].count_nonzero().item(), 0)
            self.assertEqual(window["dof_vel"].count_nonzero().item(), 0)
            self.assertTrue(torch.isfinite(observe(self.dataset, window)).all())
            torch.testing.assert_close(window["joint_rot"][:, 0], window["joint_rot"][:, -1])

    def test_joint_controls_recompute_geometry_without_modifying_reference(self):
        before = {key: value.clone() for key, value in self.reference.items()}
        changed = perturb_joints(self.dataset, self.reference, 0.6)
        expected, _ = self.dataset._kin_char_model.forward_kinematics(
            changed["root_pos"], changed["root_rot"], changed["joint_rot"])
        torch.testing.assert_close(changed["body_pos"], expected)
        self.assertFalse(torch.allclose(changed["body_pos"], before["body_pos"]))
        for key in before:
            torch.testing.assert_close(self.reference[key], before[key], rtol=0, atol=0)

    def test_noise_is_paired_and_repeatable(self):
        class NoiseOnlyModel:
            def normalize(self, x):
                return x

            def ESM_SDS_loss(self, x, t_lst):
                return torch.randn(len(x), len(t_lst)).square()

        first = torch.zeros(7, 1140)
        second = torch.ones_like(first)
        kwargs = dict(horizon=10, timesteps=[22, 15, 8], batch_size=3, repeats=3, seed=37)
        left, left_repeat = score_windows(NoiseOnlyModel(), first, **kwargs)
        right, right_repeat = score_windows(NoiseOnlyModel(), second, **kwargs)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(left_repeat, right_repeat)


if __name__ == "__main__":
    unittest.main()
