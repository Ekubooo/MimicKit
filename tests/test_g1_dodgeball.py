"""G1 dodgeball data, budget and actual-Newton integration contracts."""
import contextlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import torch
import yaml
from tools.g1_dodgeball.workflow import (ENV, PRIOR_ENV, AGENT, ENGINE, PRIOR_CFG, DATASET,
    BUDGETS, FEATURE_KEYS, prepare_motions, audit_assets, check_prior, validate_training_output)


class G1DodgeballContracts(unittest.TestCase):
    def test_zero_exit_with_nonfinite_or_constraint_overflow_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            p = Path(folder)
            (p / "runtime.txt").write_text("EXIT_CODE=0\n")
            (p / "console.txt").write_text("normal\n")
            (p / "log.txt").write_text("Iteration Loss\n0 nan\n")
            self.assertFalse(validate_training_output(p))
            (p / "log.txt").write_text("Iteration Loss\n0 0.5\n")
            (p / "console.txt").write_text("nefc overflow - please increase njmax to 592\n")
            self.assertFalse(validate_training_output(p))
            (p / "console.txt").write_text("normal\n")
            self.assertTrue(validate_training_output(p))

    def test_public_parameters_and_exact_budget(self):
        old = yaml.safe_load((ROOT / "tools/diffusion_model/config/tinymdm_multi_clip.yaml").read_text())
        new = yaml.safe_load((ROOT / PRIOR_CFG).read_text())
        for key in old:
            if key not in ("env_config", "motion_file"):
                self.assertEqual(old[key], new[key], key)
        agent = yaml.safe_load((ROOT / AGENT).read_text())
        public = yaml.safe_load((ROOT / "data/agents/smp_task_humanoid_agent.yaml").read_text())
        for key in public:
            if key not in ("smp_prior_cfg", "smp_prior_model"):
                self.assertEqual(public[key], agent[key], key)
        env, pe = [yaml.safe_load((ROOT / p).read_text()) for p in (ENV, PRIOR_ENV)]
        engine = yaml.safe_load((ROOT / ENGINE).read_text())
        self.assertEqual(engine["newton_njmax"], 1024)
        self.assertEqual(engine["newton_nconmax"], 256)
        for key in FEATURE_KEYS:
            self.assertEqual(env[key], pe[key])
        self.assertFalse(env["global_obs"])
        self.assertEqual(env["proj_target_body"], "torso_link")
        self.assertEqual(env["contact_bodies"], ["left_ankle_roll_link", "right_ankle_roll_link"])
        ref = json.loads((ROOT / "tools/g1_dodgeball/reference_budget.json").read_text())
        self.assertEqual(BUDGETS["full"][1], ref["rl_reference"]["last_logged_samples"])
        self.assertEqual(BUDGETS["full"][1], 2048 * 32 * 20402)
        self.assertEqual(BUDGETS["pilot"][1], 2048 * 32 * 400)

    def test_real_motion_projection_and_prior_guard(self):
        with contextlib.chdir(ROOT):
            old_threads = torch.get_num_threads()
            torch.set_num_threads(2)
            try:
                source = Path("data/motions/g1/g1_run.pkl").read_bytes()
                report = audit_assets()
                self.assertEqual(source, Path("data/motions/g1/g1_run.pkl").read_bytes())
                self.assertEqual(report["prior_feature_size"], 2100)
                self.assertLess(report["max_joint_limit_excess_rad"], 1e-5)
                walk, run = report["motion_preparation"]
                self.assertEqual(walk["changes"], [])
                self.assertEqual(len(run["changes"]), 2)
                self.assertEqual([x["changed_frames"] for x in run["changes"]], [8, 8])
                with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
                    p = Path(folder)
                    cfg = yaml.safe_load(Path(PRIOR_CFG).read_text())
                    cfg.update(env_config=str(p / "env_config.yaml"), num_iterations=2)
                    (p / "env_config.yaml").write_text(Path(PRIOR_ENV).read_text())
                    (p / "diffusion_config.yaml").write_text(yaml.safe_dump(cfg))
                    (p / "motion_audit.json").write_text(json.dumps(report))
                    torch.save({"ema_dmodel.step": torch.tensor(1)}, p / "model.pt")
                    with self.assertRaises(ValueError):
                        check_prior(p, report)
                    torch.save({"ema_dmodel.step": torch.tensor(2)}, p / "model.pt")
                    self.assertEqual(check_prior(p, report)["verified_ema_step"], 2)
                    changed = dict(report, char_sha256="changed")
                    with self.assertRaises(ValueError):
                        check_prior(p, changed)
            finally:
                torch.set_num_threads(old_threads)

    def test_projection_does_not_overwrite_user_derivative(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            with contextlib.chdir(folder):
                Path("data/assets/g1").mkdir(parents=True)
                Path("data/assets/g1/g1.xml").write_bytes((ROOT / "data/assets/g1/g1.xml").read_bytes())
                Path("data/datasets").mkdir(parents=True)
                Path(DATASET).write_bytes((ROOT / DATASET).read_bytes())
                Path("data/motions/g1").mkdir(parents=True)
                for name in ("walk", "run"):
                    p = Path(f"data/motions/g1/g1_{name}.pkl")
                    p.write_bytes((ROOT / p).read_bytes())
                report = prepare_motions()
                target = Path(report[1]["derived"])
                target.write_bytes(b"preserve user edits")
                with self.assertRaises(FileExistsError):
                    prepare_motions()
                self.assertEqual(target.read_bytes(), b"preserve user edits")


@unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON") == "1", "Opt in to real Newton.")
class G1DodgeballNewtonContracts(unittest.TestCase):
    def test_ball_launch_features_and_dynamic_kinematics(self):
        from envs.env_builder import build_env
        from tools.diffusion_model.motion_prior_dataset import MotionPriorData
        with contextlib.chdir(ROOT):
            prepare_motions()
            env = build_env(ENV, ENGINE, 4, "cuda:0", False)
            try:
                obs, info = env.reset()
                self.assertEqual(env._engine._solver_njmax, 1024)
                data = MotionPriorData(yaml.safe_load(Path(PRIOR_CFG).read_text()), "cuda:0")
                expected = data._compute_smp_obs_demo(env._motion_ids, env._get_motion_times())
                # Local-heading features subtract translated float32 world poses.
                # Match the engine FK accuracy budget, rather than requiring bitwise translation invariance.
                torch.testing.assert_close(info["disc_obs"], expected, rtol=1e-5, atol=1e-4)
                char = env._get_char_id()
                # A near launch must occur; verify an actual moving projectile, not just configuration.
                env._proj_trigger_times.fill_(0)
                env.step(env._engine.get_dof_pos(char).clone())
                _, ball_vel = env._get_proj_states()
                self.assertTrue(bool((ball_vel.norm(dim=-1) > 10).all()))
                self.assertTrue(bool((env._proj_trigger_times > 0).all()))
                # Validate GSI decoding/reset using genuine reference features, independently of a toy prior's quality.
                env.init_gsi_buffer(expected.reshape(4, 10, 210))
                obs, info = env.reset()
                self.assertTrue(bool(torch.isfinite(obs).all() & torch.isfinite(info["disc_obs"]).all()))
                for _ in range(8):
                    env.step(env._engine.get_dof_pos(char).clone())
                    pos, _ = env._kin_char_model.forward_kinematics(env._engine.get_root_pos(char),
                        env._engine.get_root_rot(char), env._kin_char_model.dof_to_rot(env._engine.get_dof_pos(char)))
                    torch.testing.assert_close(env._engine.get_body_pos(char), pos, atol=2e-4, rtol=1e-4)
                    self.assertTrue(bool(torch.isfinite(env._disc_obs_buf).all()))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
