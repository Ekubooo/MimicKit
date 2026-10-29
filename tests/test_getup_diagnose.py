"""Contracts for frozen reward decomposition and read-only trajectory scoring."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

import torch
from envs.task_getup_env import compute_getup_reward
from tools.getup.diagnose_getup import reward_components, score_windows, summarize_rows


class DiagnosticContracts(unittest.TestCase):
    def test_components_match_actual_task_reward_both_modes(self):
        height = torch.tensor([0.1, 0.59, 0.6, 1.2, 1.5])
        velocity = torch.tensor([-.4, 0., .25, .4, 1.])
        for mode in ("deficit", "paper_literal"):
            env = SimpleNamespace(_target_height=1.2, _target_speed=.25, _low_height=.6,
                                  _height_scale=6., _speed_scale=100., _reward_mode=mode)
            h, v = reward_components(height, velocity, env)
            actual = compute_getup_reward(height, velocity, 1.2, .6, .25, 6., 100., mode == "paper_literal")
            torch.testing.assert_close(.3 * h + .7 * v, actual)

    def test_saved_normalization_and_mean_of_exponential_without_rng_mutation(self):
        class Prior:
            def normalize(self, x): return x
            def ESM_SDS_loss(self, x, t_lst): return torch.rand(len(x), len(t_lst)) * 2
        class Normalizer:
            def get_count(self): return torch.tensor([100])
            def normalize(self, x): return x / torch.tensor([2., 3., 4.])
        agent = SimpleNamespace(_device="cpu", _env=SimpleNamespace(_num_disc_obs_steps=10),
                                _prior_model=Prior(), _sds_normalizer=Normalizer(),
                                _diffusion_steps=[22, 15, 8], _sds_loss_scale=1.2, _smp_reward_scale=.7)
        torch.manual_seed(17)
        before = torch.random.get_rng_state().clone()
        raw, normalized, reward = score_windows(agent, torch.zeros(5, 1140), 37, repeats=3, batch_size=3)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        with torch.random.fork_rng():
            losses = []
            for seed in (37, 38, 39):
                torch.manual_seed(seed)
                losses.append(torch.rand(5, 3) * 2)
            stacked = torch.stack(losses)
        expected_norm = stacked / torch.tensor([2., 3., 4.])
        torch.testing.assert_close(raw, stacked.mean(dim=0))
        torch.testing.assert_close(normalized, expected_norm.mean(dim=0))
        torch.testing.assert_close(reward, (.7 * torch.exp(-1.2 * expected_norm.mean(dim=2))).mean(dim=0))
        self.assertFalse(torch.allclose(reward, .7 * torch.exp(-1.2 * normalized.mean(dim=1))))

    def test_motion_and_velocity_summary(self):
        keys = ("head_z", "root_z", "root_vz", "root_vz_fd", "torso_up", "right_foot_force",
                "left_foot_force", "dof_speed", "action_change", "dof_change", "height_reward",
                "velocity_reward", "task_reward", "smp_reward", "weighted_task", "weighted_smp",
                "combined_reward", "vz_fd_abs_error", "standing")
        rows = [{**dict.fromkeys(keys, 1.), "root_vz": 2., "dt": .1,
                 "hold_seconds": .3, "success": 0.} for _ in range(3)]
        result = summarize_rows(rows)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["net_root_z_change"], .3)
        self.assertEqual(result["integrated_reported_vz"], .6)
        self.assertFalse(result["success"])
        self.assertEqual(summarize_rows([]), {"steps": 0})


if __name__ == "__main__":
    unittest.main()
