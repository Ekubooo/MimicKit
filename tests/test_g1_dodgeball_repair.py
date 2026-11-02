import contextlib
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import torch
import yaml
from learning.smp_g1_dodgeball_agent import physical_target_loss, joint_limit_pressure


class RepairContracts(unittest.TestCase):
    def test_old_position_clipping_remains_opt_in(self):
        from types import SimpleNamespace
        from envs.char_env import CharEnv
        env = SimpleNamespace(_action_bound_low=torch.tensor([-4.]), _action_bound_high=torch.tensor([4.]),
            _physical_target_low=torch.tensor([0.]), _physical_target_high=torch.tensor([2.]),
            _clip_position_targets_to_physical_limits=False)
        request = torch.tensor([[-3.], [3.]])
        torch.testing.assert_close(CharEnv._clip_action(env, request), request)
        env._clip_position_targets_to_physical_limits = True
        torch.testing.assert_close(CharEnv._clip_action(env, request), torch.tensor([[0.], [2.]]))

    def test_target_loss_gradient_points_into_physical_range(self):
        q = torch.tensor([[-0.5, 0.5, 1.5]], requires_grad=True)
        loss = physical_target_loss(q, torch.zeros(3), torch.ones(3), .25)
        loss.backward()
        self.assertLess(float(q.grad[0, 0]), 0)
        self.assertEqual(float(q.grad[0, 1]), 0)
        self.assertGreater(float(q.grad[0, 2]), 0)
        self.assertEqual(float(physical_target_loss(torch.tensor([[0.2, .8]]), torch.zeros(2), torch.ones(2), .25)), 0)

    def test_pressure_is_bounded_and_does_not_penalize_interior(self):
        cost = joint_limit_pressure(torch.tensor([[.5, .5], [0., 1.], [-4., 8.]]),
                                   torch.zeros(2), torch.ones(2), .03)
        torch.testing.assert_close(cost, torch.tensor([0., 1., 1.]))

    def test_variants_only_differ_in_pressure_cost(self):
        from tools.g1_dodgeball.repair import AGENTS, BUDGETS, ENV
        a, b = [yaml.safe_load((ROOT / p).read_text()) for p in AGENTS.values()]
        self.assertEqual(a.pop("limit_pressure_weight"), 0)
        self.assertEqual(b.pop("limit_pressure_weight"), .25)
        self.assertEqual(a, b)
        self.assertEqual(BUDGETS["pilot"], (2048, 2048*32*400))
        self.assertTrue(yaml.safe_load((ROOT / ENV).read_text())["clip_position_targets_to_physical_limits"])
        self.assertNotIn("clip_position_targets_to_physical_limits",
                         yaml.safe_load((ROOT / "data/envs/smp_dodgeball_g1_env.yaml").read_text()))


@unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON") == "1", "Opt in to Newton.")
class NewtonRepairContracts(unittest.TestCase):
    def test_actual_clipped_commands_gsi_interface_update_and_reload(self):
        from envs.env_builder import build_env
        from learning.agent_builder import build_agent
        from learning.smp_agent import SMPAgent
        from tools.g1_dodgeball.repair import ENV, AGENTS
        from tools.g1_dodgeball.workflow import ENGINE, prepare_motions
        from tools.diffusion_model.motion_prior_dataset import MotionPriorData
        from tools.g1_dodgeball.joint_diagnostics import JointDiagnostics
        with contextlib.chdir(ROOT):
            prepare_motions()
            env = build_env(ENV, ENGINE, 8, "cuda:0", False)
            agent = None
            try:
                # Local toy prior is only 2k updates. Use genuine reference windows for the GSI reset/update
                # interface test, not an inaccurate generated state. Production GSI stays enabled/unmodified.
                prior_dir = Path("output/g1_dodgeball_local_check_20261031/prior_2000")
                if not (prior_dir / "model.pt").is_file():
                    self.skipTest("Local diagnostic prior unavailable; run CPU contracts instead.")
                data = MotionPriorData(yaml.safe_load((prior_dir / "diffusion_config.yaml").read_text()), "cuda:0")
                def references(self, n):
                    return data.fetch_obs_demo(n).reshape(n, 10, 210)
                cfg = yaml.safe_load(Path(AGENTS["pressure"]).read_text())
                cfg.update(smp_prior_cfg=str(prior_dir / "diffusion_config.yaml"),
                           smp_prior_model=str(prior_dir / "model.pt"), test_episodes=0)
                with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
                    p = Path(folder)
                    (p / "agent.yaml").write_text(yaml.safe_dump(cfg))
                    from util import mp_util
                    mp_util.init(0, 1, "cuda:0", 0)
                    with patch.object(SMPAgent, "_generate_init_states", references):
                        agent = build_agent(str(p / "agent.yaml"), env, "cuda:0")
                        env.reset()
                        extreme = torch.full((8, 29), -10., device="cuda:0")
                        env._apply_action(extreme)
                        sent = env._engine._controls.target_pos[env._get_char_id()]
                        self.assertTrue(bool((sent >= agent._physical_lower - 1e-6).all()))
                        self.assertTrue(bool((sent <= agent._physical_upper + 1e-6).all()))
                        diagnostic = JointDiagnostics(env)
                        diagnostic.record(env, torch.ones(8, device="cuda:0", dtype=torch.bool),
                                          extreme, torch.zeros(8, device="cuda:0"))
                        diagnostic.save(p)
                        import json
                        summary = json.loads((p / "diagnostics.json").read_text())
                        for row in summary["partitions"]["all"]["joints"]:
                            self.assertEqual(row["command_outside_physical_limits_fraction"], 0)
                        counts = agent._prior_model.obs_normalizer.get_count().clone()
                        agent.train_model(256, str(p), False, "txt")
                        self.assertTrue(bool(torch.isfinite(agent._exp_buffer.get_data_flat("reward")).all()))
                        for tensor in agent.state_dict().values():
                            self.assertTrue(bool(torch.isfinite(tensor).all()))
                        torch.testing.assert_close(agent._prior_model.obs_normalizer.get_count(), counts)
                        self.assertTrue((p / "model.pt").is_file())
                        agent.load(str(p / "model.pt"))
                        agent.close()
                        env.close()
                        mp_util.shutdown()
                        # Real CLI-equivalent saved configuration and fresh-agent evaluation/load path.
                        (p / "agent_config.yaml").write_text(yaml.safe_dump(cfg))
                        (p / "env_config.yaml").write_text(Path(ENV).read_text())
                        (p / "engine_config.yaml").write_text(Path(ENGINE).read_text())
                        from tools.g1_dodgeball.evaluate import run
                        from types import SimpleNamespace
                        evaluation = p / "evaluation"
                        evaluation.mkdir()
                        run(SimpleNamespace(model_file=str(p / "model.pt"), prior_dir=str(prior_dir),
                            device="cuda:0", init="reference", episodes=4, seed=123,
                            projectiles="on", joint_diagnostics=True), evaluation)
                        report = json.loads((evaluation / "diagnostics.json").read_text())
                        for row in report["partitions"]["all"]["joints"]:
                            self.assertEqual(row["command_outside_physical_limits_fraction"], 0)
            finally:
                if agent is not None:
                    agent.close()
                env.close()
                from util import mp_util
                mp_util.shutdown()


if __name__ == "__main__":
    unittest.main()
