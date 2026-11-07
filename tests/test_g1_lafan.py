import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
import torch
import yaml
from anim.mjcf_char_model import MJCFCharModel
from envs.g1_lafan_adapter import G1LafanAdapter, identity, rotation
from envs.amp_env import compute_disc_obs
from envs.smp_env import SMPEnv
from util import torch_util as tu


class LafanContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = MJCFCharModel("cpu")
        cls.g.load("data/assets/g1/g1.xml")
        cls.a = G1LafanAdapter(cls.g, "cpu")

    def pose(self):
        p = torch.zeros(2, 10, 3)
        p[..., 2] = .8
        r = identity((2, 10), "cpu")
        q = torch.zeros(2, 10, 29)
        for side in ("right", "left"):
            q[..., self.g.get_joint_dof_idx(self.g.get_body_id(side + "_elbow_link"))] = math.pi / 2
            q[..., self.g.get_joint_dof_idx(self.g.get_body_id(side + "_knee_link"))] = .3
        return p, r, q

    def test_anatomical_elbow_and_knee_signs(self):
        p, r, q = self.pose()
        _, j, _ = self.a.to_human(p, r, self.g.dof_to_rot(q))
        for side in ("right", "left"):
            elbow = self.a.human.get_body_id(side + "_lower_arm")
            knee = self.a.human.get_body_id(side + "_shin")
            self.assertLess(float(tu.quat_to_exp_map(j[..., elbow-1, :]).abs().max()), 1e-5)
            torch.testing.assert_close(self.a.human.get_joint(knee).rot_to_dof(j[..., knee-1, :]),
                                       torch.full((2, 10, 1), .3), atol=1e-5, rtol=0)
        q[..., self.g.get_joint_dof_idx(self.g.get_body_id("left_elbow_link"))] = 0
        _, j, _ = self.a.to_human(p, r, self.g.dof_to_rot(q))
        i = self.a.human.get_body_id("left_lower_arm")
        torch.testing.assert_close(self.a.human.get_joint(i).rot_to_dof(j[..., i-1, :]),
                                   torch.full((2, 10, 1), -math.pi/2), atol=1e-5, rtol=0)

    def test_world_translation_and_yaw_invariance(self):
        p, r, q = self.pose()
        v = torch.ones_like(p) * .1
        j = self.g.dof_to_rot(q)
        original = self.a.observations(p, r, v, v, j, 1/30)
        yaw = rotation(2, .91, "cpu").expand_as(r)
        moved = tu.quat_rotate(yaw, p) + torch.tensor([10., -7., 0.])
        changed = self.a.observations(moved, tu.quat_mul(yaw, r), tu.quat_rotate(yaw, v),
                                      tu.quat_rotate(yaw, v), j, 1/30)
        torch.testing.assert_close(original, changed, atol=1e-5, rtol=1e-5)
        self.assertEqual(original.shape, (2, 1140))

    def test_inverse_limits_clearance_roundtrip_and_velocities(self):
        p, r, q = self.pose()
        zero = torch.zeros_like(p)
        obs = self.a.observations(p, r, zero, zero, self.g.dof_to_rot(q), 1/30)
        frames = self.a.from_samples(obs.reshape(2, 10, 114))
        actual = frames[..., 6:]
        self.assertTrue(bool((actual >= self.a.low-1e-6).all() & (actual <= self.a.high+1e-6).all()))
        torch.testing.assert_close(actual, q, atol=2e-4, rtol=0)
        bp, br = self.g.forward_kinematics(frames[..., :3], tu.exp_map_to_quat(frames[..., 3:6]), self.g.dof_to_rot(actual))
        self.assertGreaterEqual(float(self.a.sole_height(bp, br, "g1").min()), .0019)
        from types import SimpleNamespace
        stub = SimpleNamespace(_kin_char_model=self.g, _device="cpu",
                               _engine=SimpleNamespace(get_timestep=lambda: 1/30))
        frames[:, :, 0] += torch.arange(10) / 30
        state = SMPEnv._motion_frames_to_init_states(stub, frames)
        torch.testing.assert_close(state["root_vel"][..., 0], torch.ones(2, 1, 10), atol=1e-5, rtol=0)
        self.assertTrue(all(bool(torch.isfinite(v).all()) for v in state.values()))

    def test_canonical_feature_order_matches_original(self):
        p, r, q = self.pose()
        v = torch.zeros_like(p)
        hp, hj, hb = self.a.to_human(p, r, self.g.dof_to_rot(q))
        reference = compute_disc_obs(hp[:, -1], r[:, -1], hp, r, v, v, hj,
                                     torch.zeros(2, 10, 28), hb[..., self.a.key_ids, :], False, True, False)
        actual = self.a.observations(p, r, v, v, self.g.dof_to_rot(q), 1/30)
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)

    def test_original_agent_parameters_preserved(self):
        original = yaml.safe_load((ROOT/"data/agents/smp_task_humanoid_agent.yaml").read_text())
        current = yaml.safe_load((ROOT/"data/agents/smp_dodgeball_g1_lafan_agent.yaml").read_text())
        for k, v in original.items():
            if k != "agent_name":
                self.assertEqual(current[k], v, k)

    def test_bad_generated_windows_fail_instead_of_becoming_fallback_motion(self):
        with self.assertRaises(ValueError):
            self.a.from_samples(torch.zeros(2, 10, 210))
        with self.assertRaises(ValueError):
            self.a.from_samples(torch.full((2, 10, 114), float("nan")))

    def test_full_budget_and_original_task_parameters(self):
        sys.path.insert(0, str(ROOT))
        from tools.g1_dodgeball.lafan import FULL_SAMPLES
        self.assertEqual(FULL_SAMPLES, 4096 * 32 * 10201)
        self.assertEqual(FULL_SAMPLES, 2048 * 32 * 20402)
        original = yaml.safe_load((ROOT/"data/envs/smp_dodgeball_humanoid_env.yaml").read_text())
        current = yaml.safe_load((ROOT/"data/envs/smp_dodgeball_g1_lafan_env.yaml").read_text())
        for k in ("episode_length", "num_projectiles", "proj_dist_min", "proj_dist_max",
                  "proj_speed_min", "proj_speed_max", "proj_trigger_time_min", "proj_trigger_time_max",
                  "enable_early_termination", "num_disc_obs_steps", "disc_dof_vel_obs"):
            self.assertEqual(current[k], original[k], k)
        self.assertEqual(current["dodge_reward_mode"], "legacy")

    def test_prior_environment_contract_uses_canonical_key_order(self):
        from learning.smp_agent import SMPAgent
        from types import SimpleNamespace
        from unittest.mock import patch, mock_open
        stub = SimpleNamespace(_env=SimpleNamespace(
            _global_obs=False, _root_height_obs=True, _enable_tar_obs=False,
            _num_disc_obs_steps=10, _disc_dof_vel_obs=False, _key_body_ids=list(range(5)),
            _lafan_adapter=self.a, _engine=SimpleNamespace(get_timestep=lambda: 1/30)))
        cfg = yaml.safe_load((ROOT/"tools/diffusion_model/config/tinymdm_multi_clip.yaml").read_text())
        SMPAgent._check_prior_env_config(stub, cfg)
        bad = yaml.safe_load((ROOT/cfg["env_config"]).read_text())
        bad["key_bodies"] = list(reversed(bad["key_bodies"]))
        with patch("builtins.open", mock_open(read_data=yaml.safe_dump(bad))):
            with self.assertRaises(ValueError):
                SMPAgent._check_prior_env_config(stub, cfg)


if __name__ == "__main__":
    unittest.main()
