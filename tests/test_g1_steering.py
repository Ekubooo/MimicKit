"""Original Steering contracts and exact-budget launch plans for the G1 port."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))


class SteeringContracts(unittest.TestCase):
    def test_original_task_and_smp_parameters_preserved(self):
        def read(path):
            return yaml.safe_load((ROOT / path).read_text())

        original = read("data/envs/smp_steering_humanoid_env.yaml")
        current = read("data/envs/smp_steering_g1_lafan_env.yaml")
        embodiment = {"env_name", "char_file", "init_pose", "key_bodies", "contact_bodies", "motion_file"}
        for key, value in original.items():
            if key not in embodiment:
                self.assertEqual(current[key], value, key)
        original = read("data/agents/smp_task_humanoid_agent.yaml")
        current = read("data/agents/smp_steering_g1_lafan_agent.yaml")
        for key, value in original.items():
            if key != "agent_name":
                self.assertEqual(current[key], value, key)
        # Task reward is inherited directly: the adapter only changes prior observations/GSI.
        from envs.task_steering_env import TaskSteeringEnv
        from envs.task_steering_g1_lafan_env import TaskSteeringG1LafanEnv
        self.assertIs(TaskSteeringG1LafanEnv._update_reward, TaskSteeringEnv._update_reward)
        self.assertIs(TaskSteeringG1LafanEnv._compute_obs, TaskSteeringEnv._compute_obs)

    def test_full_and_resource_fallback_match_published_sample_budget(self):
        log = (ROOT / "data/logs/smp_steering_humanoid_log.txt").read_text().splitlines()
        last = dict(zip(log[0].split(), log[-1].split()))
        published_samples = int(last["Samples"])
        for n, updates in ((4096, 17301), (2048, 34602)):
            result = subprocess.run([
                sys.executable, "-B", "-m", "tools.g1_steering.workflow", "--stage", "full",
                "--num_envs", str(n), "--out_dir", "output/steering_dry_run", "--dry_run",
            ], cwd=ROOT, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["task"], "steering")
            self.assertEqual(plan["updates"], updates)
            self.assertEqual(plan["max_samples"], published_samples)
            self.assertEqual(n * 32 * updates, published_samples)
            self.assertNotIn("--model_file", plan["command"])
            self.assertIn("data/envs/smp_steering_g1_lafan_env.yaml", plan["command"])


if __name__ == "__main__":
    unittest.main()
