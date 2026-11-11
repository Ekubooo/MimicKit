"""Original Location contracts and exact-budget launch plans for the G1 port."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))


class LocationContracts(unittest.TestCase):
    def test_original_task_and_smp_parameters_preserved(self):
        def read(path):
            return yaml.safe_load((ROOT / path).read_text())

        original = read("data/envs/smp_location_humanoid_env.yaml")
        current = read("data/envs/smp_location_g1_lafan_env.yaml")
        embodiment = {"env_name", "char_file", "init_pose", "key_bodies", "contact_bodies", "motion_file"}
        for key, value in original.items():
            if key not in embodiment:
                self.assertEqual(current[key], value, key)
        original = read("data/agents/smp_task_humanoid_agent.yaml")
        current = read("data/agents/smp_location_g1_lafan_agent.yaml")
        for key, value in original.items():
            if key != "agent_name":
                self.assertEqual(current[key], value, key)
        # Keep the author's task implementation, including target sampling and scheduling.
        from envs.task_location_env import TaskLocationEnv
        from envs.task_location_g1_lafan_env import TaskLocationG1LafanEnv
        for method in ("_update_reward", "_compute_obs", "_reset_tar", "_update_task"):
            self.assertIs(getattr(TaskLocationG1LafanEnv, method), getattr(TaskLocationEnv, method))
        from envs.g1_lafan_env import G1LafanEnvMixin
        self.assertTrue(issubclass(TaskLocationG1LafanEnv, G1LafanEnvMixin))

    def test_full_and_resource_fallback_match_published_sample_budget(self):
        log = (ROOT / "data/logs/smp_location_humanoid_log.txt").read_text().splitlines()
        last = dict(zip(log[0].split(), log[-1].split()))
        published_samples = int(last["Samples"])
        args = shlex.split((ROOT / "args/smp_location_g1_lafan_args.txt").read_text())
        defaults = dict(zip(args[::2], args[1::2]))
        self.assertEqual(int(defaults["--max_samples"]), published_samples)
        self.assertEqual(defaults["--num_envs"], "4096")
        for n, updates in ((4096, 17401), (2048, 34802)):
            result = subprocess.run([
                sys.executable, "-B", "-m", "tools.g1_location.workflow", "--stage", "full",
                "--num_envs", str(n), "--out_dir", "output/location_dry_run", "--dry_run",
            ], cwd=ROOT, capture_output=True, text=True, check=True)
            plan = json.loads(result.stdout)
            self.assertEqual(plan["task"], "location")
            self.assertEqual(plan["engine"], "newton_1_7")
            self.assertEqual(plan["num_envs"], n)
            self.assertEqual(plan["updates"], updates)
            self.assertEqual(plan["max_samples"], published_samples)
            self.assertEqual(n * 32 * updates, published_samples)
            self.assertNotIn("--model_file", plan["command"])
            self.assertIn(defaults["--env_config"], plan["command"])
            self.assertIn(defaults["--engine_config"], plan["command"])
        self.assertEqual(17401, int(last["Iteration"]) + 1)


if __name__ == "__main__":
    unittest.main()
