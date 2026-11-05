import contextlib
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import torch
import yaml
from tools.g1_dodgeball.limit_training import build_configs, BUDGETS
from envs.char_env import CharEnv
from envs.task_dodgeball_env import compute_paper_dodge_reward, compute_dodge_reward
from learning.smp_g1_dodgeball_agent import physical_target_loss


class LimitContracts(unittest.TestCase):
    def test_paper_reward_has_negative_near_ball_cost_and_no_velocity_term(self):
        root = torch.zeros(3, 3)
        ball = torch.tensor([[[0., 0., 0.]], [[1.5, 0., 0.]], [[10., 0., 0.]]])
        actual = compute_paper_dodge_reward(root, ball)
        torch.testing.assert_close(actual, torch.tensor([1-math.exp(.45), 0., 1-math.exp(-2.55)]))
        self.assertGreater(float(compute_dodge_reward(root, root, ball)[0]), 0)

    def test_targets_make_position_feedback_point_inward_at_both_stops(self):
        # Representative G1 knee: the physical lower stop remains negative.
        env = SimpleNamespace(_action_bound_low=torch.tensor([-4.]), _action_bound_high=torch.tensor([4.]),
                              _physical_target_low=torch.tensor([.03]), _physical_target_high=torch.tensor([2.8498]),
                              _clip_position_targets_to_physical_limits=True)
        target = CharEnv._clip_action(env, torch.tensor([[-100.], [100.]]))
        q = torch.tensor([[-.087267], [2.8798]])
        self.assertGreater(float((target-q)[0]), 0)
        self.assertLess(float((target-q)[1]), 0)
        raw = torch.tensor([[-.04]], requires_grad=True)
        physical_target_loss(raw, env._physical_target_low, env._physical_target_high, .25).backward()
        self.assertLess(float(raw.grad[0]), 0)

    def test_invalid_margin_rejected_before_engine_creation(self):
        base, _ = build_configs()
        for margin in [-.01, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                CharEnv(dict(base, position_target_margin_rad=margin), {}, 1, "cpu", False)

    def test_original_configs_and_full_budget(self):
        env, agent = build_configs()
        self.assertEqual((agent["task_reward_weight"], agent["smp_reward_weight"], agent["sds_loss_scale"]), (.6, 1.2, 4))
        self.assertEqual(agent["limit_pressure_weight"], 0)
        self.assertNotIn("dodge_reward_mode", yaml.safe_load((ROOT/"data/envs/smp_dodgeball_g1_env.yaml").read_text()))
        self.assertEqual(BUDGETS["full"][0]*32*BUDGETS["full"][1], 1337065472)


@unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON") == "1", "Opt in to local Newton contract.")
class NewtonLimitContracts(unittest.TestCase):
    def test_engine_commands_training_reload_and_saved_config_evaluation(self):
        from envs.env_builder import build_env
        from learning.agent_builder import build_agent
        from learning.smp_agent import SMPAgent
        from tools.diffusion_model.motion_prior_dataset import MotionPriorData
        from tools.g1_dodgeball.workflow import ENGINE, prepare_motions
        from tools.g1_dodgeball.evaluate import run
        from util import mp_util
        import json
        with contextlib.chdir(ROOT):
            prior = Path("output/g1_dodgeball_local_check_20261031/prior_2000")
            if not (prior/"model.pt").is_file():
                self.skipTest("Local toy prior missing; real-prior smoke is required remotely.")
            prepare_motions()
            env_cfg, agent_cfg = build_configs()
            agent_cfg.update(smp_prior_cfg=str(prior/"diffusion_config.yaml"),
                             smp_prior_model=str(prior/"model.pt"), test_episodes=0)
            data = MotionPriorData(yaml.safe_load((prior/"diffusion_config.yaml").read_text()), "cuda:0")
            # Local toy-prior test exercises the GSI interface with reference states.
            # Production must separately smoke-test the real frozen prior and actual GSI.
            def references(self, n):
                return data.fetch_obs_demo(n).reshape(n, 10, 210)
            with tempfile.TemporaryDirectory(dir=ROOT/"output") as folder:
                out = Path(folder)
                for name, cfg in (("env_config.yaml", env_cfg), ("agent_config.yaml", agent_cfg)):
                    (out/name).write_text(yaml.safe_dump(cfg))
                (out/"engine_config.yaml").write_text(Path(ENGINE).read_text())
                env = agent = None
                try:
                    mp_util.init(0, 1, "cuda:0", 0)
                    env = build_env(str(out/"env_config.yaml"), ENGINE, 8, "cuda:0", False)
                    with patch.object(SMPAgent, "_generate_init_states", references):
                        agent = build_agent(str(out/"agent_config.yaml"), env, "cuda:0")
                        for value in [-100., 100.]:
                            env._apply_action(torch.full((8, 29), value, device="cuda:0"))
                            sent = env._engine._controls.target_pos[env._get_char_id()]
                            self.assertTrue(bool((sent >= env._physical_target_low-1e-6).all()))
                            self.assertTrue(bool((sent <= env._physical_target_high+1e-6).all()))
                        for name in ("left_knee_joint", "right_knee_joint"):
                            joint = next(env._kin_char_model.get_joint(i) for i in range(1, env._kin_char_model.get_num_joints())
                                         if env._kin_char_model.get_joint(i).name == name)
                            self.assertAlmostEqual(float(env._physical_target_low[joint.dof_idx]), .03, places=6)
                        agent.train_model(768, str(out), False, "txt")
                        from tools.g1_dodgeball.style_balance import validate_log
                        validate_log(out/"log.txt", 768, 3)
                        agent.load(str(out/"model.pt"))
                        self.assertTrue(all(bool(torch.isfinite(t).all()) for t in agent.state_dict().values()))
                    agent.close(); agent = None
                    env.close(); env = None
                    mp_util.shutdown()
                    evaluation = out/"evaluation"
                    evaluation.mkdir()
                    run(SimpleNamespace(model_file=str(out/"model.pt"), prior_dir=str(prior), device="cuda:0",
                                        init="reference", episodes=4, seed=123, projectiles="on", joint_diagnostics=True), evaluation)
                    report = json.loads((evaluation/"diagnostics.json").read_text())
                    for joint in report["partitions"]["all"]["joints"]:
                        self.assertEqual(joint["command_outside_physical_limits_fraction"], 0)
                finally:
                    if agent is not None:
                        agent.close()
                    if env is not None:
                        env.close()
                    mp_util.shutdown()


if __name__ == "__main__":
    unittest.main()
