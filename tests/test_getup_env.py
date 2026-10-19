"""Getup reward/reset/metric contracts; opt-in real Newton integration below."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

import torch
import yaml

from envs.base_env import DoneFlags, EnvMode
from envs.getup_util import GroundSupports
from envs.task_getup_env import TaskGetupEnv, compute_getup_reward
from learning.base_agent import AgentMode
from learning.smp_agent import SMPAgent
from learning.smp_getup_agent import SMPGetupAgent
from tools.getup.common import FULL_SAMPLES


class GetupContracts(unittest.TestCase):
    def reward(self, height, velocity, literal=False):
        return compute_getup_reward(torch.tensor(height), torch.tensor(velocity),
                                    1.2, 0.6, 0.25, 6.0, 100.0, literal)

    def test_height_and_upward_speed_reward_direction(self):
        height = self.reward([0.1, 0.4, 0.8, 1.2, 1.5], [0.25] * 5)
        self.assertTrue(bool((height[1:] >= height[:-1]).all()))
        self.assertAlmostEqual(float(height[-1]), 1.0)
        velocity = self.reward([0.2] * 4, [-0.25, 0.0, 0.25, 1.0])
        self.assertTrue(bool((velocity[1:] >= velocity[:-1]).all()))
        self.assertTrue(bool(((velocity >= 0) & (velocity <= 1)).all()))
        # The printed equation has the opposite direction; preserve explicitly.
        literal = self.reward([0.2] * 4, [-0.25, 0.0, 0.25, 1.0], literal=True)
        self.assertGreater(float(literal[0]), float(literal[-1]))
        self.assertFalse(torch.allclose(literal, velocity))

    def test_geometry_supports_include_offsets_radius_and_box_corners(self):
        class Kin:
            def get_body_id(self, name):
                return 0
        xml = '''<mujoco><worldbody><body name="pelvis">
        <geom type="sphere" pos="0 0 -0.1" size="0.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.1"/>
        <geom type="box" pos="0 0 -0.5" size="0.1 0.2 0.3"/>
        </body></worldbody></mujoco>'''
        (ROOT / "output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            path = Path(folder) / "asset.xml"
            path.write_text(xml)
            support = GroundSupports(path, Kin(), "cpu")
        q = torch.tensor([[[0., 0., 0., 1.]]])
        z = support.minimum_height(torch.zeros(1, 1, 3), q)
        torch.testing.assert_close(z, torch.tensor([-0.8]))
        q = torch.tensor([[[1., 0., 0., 0.]]])  # Upside down.
        z = support.minimum_height(torch.zeros(1, 1, 3), q)
        torch.testing.assert_close(z, torch.tensor([-0.1]))

    def test_budget_and_feature_configuration(self):
        self.assertEqual(FULL_SAMPLES, 2048 * 32 * 15259)
        env = yaml.safe_load((ROOT / "data/envs/smp_getup_humanoid_env.yaml").read_text())
        prior = yaml.safe_load((ROOT / "data/envs/smp_getup_prior_env.yaml").read_text())
        for key in prior:
            self.assertEqual(env[key], prior[key], key)
        agent = yaml.safe_load((ROOT / "data/agents/smp_getup_humanoid_agent.yaml").read_text())
        self.assertEqual(agent["task_reward_weight"], 0.2)
        self.assertEqual(agent["smp_reward_weight"], 0.8)
        self.assertEqual(agent["sds_loss_scale"], 8)
        self.assertFalse(agent["enable_gsi"])

    def test_training_mode_restored_before_post_evaluation_reset(self):
        stub = SMPGetupAgent.__new__(SMPGetupAgent)
        torch.nn.Module.__init__(stub)
        stub._mode = AgentMode.TRAIN
        stub.set_mode = lambda mode: setattr(stub, "_mode", mode)
        # Exercise the actual try/finally override without constructing a prior.
        def evaluate(agent, count):
            agent._mode = AgentMode.TEST
            return {"num_eps": count}
        with patch.object(SMPAgent, "test_model", evaluate):
            self.assertEqual(SMPGetupAgent.test_model(stub, 8), {"num_eps": 8})
        self.assertEqual(stub._mode, AgentMode.TRAIN)


class MetricEngine:
    def __init__(self):
        self.pos = torch.tensor([[0., 0., 0.9]])
        self.vel = torch.zeros(1, 3)
        self.rot = torch.tensor([[0., 0., 0., 1.]])
        self.bodies = torch.tensor([[[0., 0., 1.3], [0., 0., 1.1], [0., 0., 0.05]]])
        self.rotations = self.rot[:, None].expand(1, 3, 4)
    def get_timestep(self): return 1 / 30
    def enabled_record_video(self): return False
    def get_root_pos(self, _): return self.pos
    def get_root_rot(self, _): return self.rot
    def get_root_vel(self, _): return self.vel
    def get_root_ang_vel(self, _): return self.vel
    def get_dof_pos(self, _): return torch.zeros(1, 28)
    def get_dof_vel(self, _): return torch.zeros(1, 28)
    def get_body_pos(self, _): return self.bodies
    def get_body_rot(self, _): return self.rotations
    def get_ground_contact_forces(self, _):
        return torch.tensor([[[0., 0., 0.], [0., 0., 0.], [0., 0., 100.]]])


class GetupMetrics(unittest.TestCase):
    def setUp(self):
        self.env = TaskGetupEnv.__new__(TaskGetupEnv)
        e = self.env
        e._engine = MetricEngine()
        e._device = "cpu"
        e._get_char_id = lambda: 0
        e.get_num_envs = lambda: 1
        e._head_id, e._torso_id, e._foot_ids = 0, 1, [2]
        e._target_height, e._low_height, e._target_speed = 1.2, .6, .25
        e._height_scale, e._speed_scale, e._reward_mode = 6., 100., "deficit"
        e._success_root_height, e._success_upright = .75, .8
        e._success_linear_speed, e._success_angular_speed, e._hold_target = .5, 1., 5.
        e._standing = torch.zeros(1, dtype=torch.bool)
        e._hold, e._best_hold = torch.zeros(1), torch.zeros(1)
        e._ever_success = torch.zeros(1, dtype=torch.bool)
        e._first_success = torch.full((1,), -1.)
        e._reward_buf, e._time_buf = torch.zeros(1), torch.zeros(1)
        e._done_buf = torch.zeros(1, dtype=torch.int)
        e._episode_length, e._max_distance = 20., 20.
        e._mode, e._totals = EnvMode.TRAIN, torch.zeros(2, 9)
        e._init_augmented = torch.zeros(1, dtype=torch.bool)
        e._initial_head = torch.tensor([.2])
        e._diagnostics = {}
        e._log_tracking_error = False

    def test_success_requires_continuous_hold_and_does_not_end_episode(self):
        e = self.env
        for i in range(150):
            e._time_buf[:] = (i + 1) / 30
            e._update_reward()
        self.assertTrue(e._ever_success.item())
        self.assertAlmostEqual(e._first_success.item(), 5.)
        e._update_done()
        self.assertEqual(e._done_buf.item(), DoneFlags.NULL.value)
        # Falling afterwards retains ever-success but clears the current hold.
        e._engine.bodies[:, 0, 2] = .2
        e._engine.pos[:, 2] = .2
        e._update_reward()
        e._update_done()
        self.assertTrue(e._ever_success.item())
        self.assertEqual(e._hold.item(), 0.)
        self.assertEqual(e._done_buf.item(), DoneFlags.NULL.value)
        e._time_buf[:] = 20.
        e._update_done()
        report = e.record_diagnostics()
        self.assertEqual(report["getup_train_success_rate"].item(), 1.)
        self.assertEqual(report["getup_train_end_rate"].item(), 0.)
        self.assertTrue(all(len(key) < 25 for key in report))
        # Diagnostic reads must not erase data before a sparse saved log row.
        self.assertEqual(e.record_diagnostics()["getup_train_episodes"].item(), 1.)
        e.clear_train_diagnostics()
        self.assertEqual(e.record_diagnostics()["getup_train_episodes"].item(), 0.)

    def test_interrupted_hold_and_nonfinite_failure(self):
        e = self.env
        e._hold[:] = 4.9
        e._engine.vel[:, 0] = 2.
        e._update_reward()
        self.assertEqual(e._hold.item(), 0.)
        self.assertFalse(e._ever_success.item())
        e._engine.pos[:, 0] = float("nan")
        e._update_done()
        self.assertEqual(e._done_buf.item(), DoneFlags.FAIL.value)
        report = e.record_diagnostics()
        self.assertEqual(report["getup_train_nan_failures"].item(), 1.)


@unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON") == "1", "Opt in to real CUDA/Newton checks.")
class GetupNewtonIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from envs import env_builder
        cls.env = env_builder.build_env("data/envs/smp_getup_humanoid_env.yaml",
                                        "data/engines/newton_engine.yaml", 8, "cuda:0", False)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_augmented_grounding_history_and_partial_resets(self):
        e = self.env
        e.set_mode(EnvMode.TEST)
        e._test_init = "augmented"
        torch.manual_seed(123)
        for _ in range(20):
            obs, info = e.reset()
            self.assertTrue(bool(torch.isfinite(obs).all()))
            self.assertTrue(bool(torch.isfinite(info["disc_obs"]).all()))
            char = e._get_char_id()
            pos, rot = e._engine.get_body_pos(char), e._engine.get_body_rot(char)
            self.assertGreaterEqual(float(e._supports.minimum_height(pos, rot).min()), .0049)
            self.assertTrue(bool((pos[:, e._head_id, 2] < .6).all()))
            hist = e._disc_hist_root_pos.get_all()
            torch.testing.assert_close(hist[:, -1], e._engine.get_root_pos(char))
            torch.testing.assert_close(hist[:, 0], hist[:, -1])
        before = e._engine.get_root_pos(char).clone()
        e.reset(torch.tensor([1, 3], device="cuda:0"))
        torch.testing.assert_close(before[[0, 2, 4, 5, 6, 7]],
                                   e._engine.get_root_pos(char)[[0, 2, 4, 5, 6, 7]])
        # Advance the global circular-buffer head, then verify partial fill.
        e.step(e._engine.get_dof_pos(char).clone())
        e.reset(torch.tensor([1, 3], device="cuda:0"))
        hist = e._disc_hist_root_pos.get_all()
        torch.testing.assert_close(hist[[1, 3], 0], hist[[1, 3], -1])

    def test_training_distribution_and_reference_history(self):
        e = self.env
        e.set_mode(EnvMode.TRAIN)
        e._aug_probability = .1
        count = 0
        torch.manual_seed(55)
        for _ in range(100):
            e.reset()
            count += int(e._init_augmented.sum())
            hist = e._disc_hist_root_pos.get_all()
            torch.testing.assert_close(hist[:, -1], e._engine.get_root_pos(e._get_char_id()))
        self.assertGreater(count, 40)
        self.assertLess(count, 120)


if __name__ == "__main__":
    unittest.main()
