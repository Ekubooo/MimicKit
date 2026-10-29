"""G1 feature, reference-limit preparation and public-baseline contracts."""
import contextlib
import os
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import numpy as np
import torch
import yaml
from tools.g1_spinkick.workflow import BUDGETS, prepare_motion, audit_assets, check_prior, ENV, PRIOR_CFG


class PreparationContracts(unittest.TestCase):
    def test_partial_prior_cannot_start_rl(self):
        (ROOT / "output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            cfg = yaml.safe_load((ROOT / PRIOR_CFG).read_text())
            cfg.update(env_config=str(folder / "env_config.yaml"), num_iterations=2)
            (folder / "env_config.yaml").write_text((ROOT / ENV).read_text())
            (folder / "diffusion_config.yaml").write_text(yaml.safe_dump(cfg))
            with contextlib.chdir(ROOT):
                torch.save({"ema_dmodel.step": torch.tensor(1)}, folder / "model.pt")
                with self.assertRaises(ValueError):
                    check_prior(folder)
                torch.save({"ema_dmodel.step": torch.tensor(2)}, folder / "model.pt")
                self.assertEqual(check_prior(folder)["verified_ema_step"], 2)

    def test_joint_projection_preserves_source_and_rejects_edited_derivative(self):
        (ROOT / "output").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            with contextlib.chdir(folder):
                for p in ("data/assets/g1", "data/motions/g1", "data/envs"):
                    Path(p).mkdir(parents=True)
                joints = ''.join(f'<joint name="j{i}" range="-.1 .1"/>' for i in range(29))
                Path("data/assets/g1/g1.xml").write_text('<mujoco><worldbody><body>'+joints+'</body></worldbody></mujoco>')
                Path("data/envs/view_motion_spinkick_g1_env.yaml").write_text('env_name: view_motion\n')
                frames = np.zeros((2, 35))
                frames[:, 6] = [.4, -.3]
                source = Path("data/motions/g1/g1_spinkick.pkl")
                source.write_bytes(pickle.dumps({"frames": frames.tolist(), "fps": 60, "loop_mode": 1}))
                before = source.read_bytes()
                report = prepare_motion()
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(report["changes"][0]["changed_frames"], 2)
                derived = Path(report["derived"])
                with derived.open('rb') as stream:
                    data = pickle.load(stream)
                np.testing.assert_allclose(np.asarray(data["frames"])[:, 6], [.1, -.1])
                self.assertEqual(prepare_motion()["derived_sha256"], report["derived_sha256"])
                derived.write_bytes(b'edited data to preserve')
                with self.assertRaises(FileExistsError):
                    prepare_motion()

    def test_baseline_parameters_and_budgets(self):
        old = yaml.safe_load((ROOT / "tools/diffusion_model/config/tinymdm_single_clip.yaml").read_text())
        new = yaml.safe_load((ROOT / PRIOR_CFG).read_text())
        for key in old:
            if key not in ("env_config", "motion_file"):
                self.assertEqual(old[key], new[key], key)
        env = yaml.safe_load((ROOT / ENV).read_text())
        self.assertEqual(env["contact_bodies"], ["left_ankle_roll_link", "right_ankle_roll_link"])
        for stage, updates in (("rl-smoke", 1), ("pilot", 400), ("full", 5802)):
            n, samples = BUDGETS[stage]
            self.assertEqual(samples, n * 32 * updates)
        import json
        reference = json.loads((ROOT / "tools/g1_spinkick/reference_budget.json").read_text())
        self.assertEqual(new["num_iterations"], reference["priors"]["spinkick"]["ema_step"])
        self.assertEqual(new["num_samples_stat"] * 10,
                         reference["priors"]["spinkick"]["normalizer_frame_count"])
        self.assertEqual(BUDGETS["full"][1], reference["rl_reference"]["last_logged_samples"])


@unittest.skipUnless((ROOT / "data/motions/g1/g1_spinkick.pkl").is_file(), "Download G1 data first.")
class G1FeatureContracts(unittest.TestCase):
    def test_real_asset_dimensions_and_motion_limits(self):
        old = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with contextlib.chdir(ROOT):
                report = audit_assets()
            self.assertEqual((report["dof"], report["frames"], report["prior_feature_size"]), (29, 78, 2100))
            self.assertLess(report["max_joint_limit_excess_rad"], 1e-5)
            changes = report["motion_preparation"]["changes"]
            self.assertEqual(changes[0]["joint"], "left_ankle_roll_joint")
            self.assertEqual(changes[0]["changed_frames"], 78)
        finally:
            torch.set_num_threads(old)


@unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON") == "1", "Opt in to real Newton checks.")
class G1NewtonContracts(unittest.TestCase):
    def test_dynamic_kinematics_and_prior_feature_alignment(self):
        from envs.env_builder import build_env
        from tools.diffusion_model.motion_prior_dataset import MotionPriorData
        with contextlib.chdir(ROOT):
            prepare_motion()
            env = build_env(ENV, "data/engines/newton_engine.yaml", 4, "cuda:0", False)
            try:
                obs, info = env.reset()
                data = MotionPriorData(yaml.safe_load(Path(PRIOR_CFG).read_text()), "cuda:0")
                expected = data._compute_smp_obs_demo(env._motion_ids, env._get_motion_times())
                torch.testing.assert_close(info["disc_obs"], expected, rtol=1e-5, atol=1e-5)
                char = env._get_char_id()
                for _ in range(8):
                    env.step(env._engine.get_dof_pos(char).clone())
                    pos, rot = env._kin_char_model.forward_kinematics(
                        env._engine.get_root_pos(char), env._engine.get_root_rot(char),
                        env._kin_char_model.dof_to_rot(env._engine.get_dof_pos(char)))
                    torch.testing.assert_close(env._engine.get_body_pos(char), pos, atol=2e-4, rtol=1e-4)
                    self.assertTrue(bool(torch.isfinite(env._disc_obs_buf).all()))
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
