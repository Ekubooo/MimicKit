"""Two bounded standing-learning controls, followed by separately labelled tests.

Run after synchronization. This does not auto-start a full get-up curriculum.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.getup.common import preflight, write_json

ENV = "data/envs/smp_getup_humanoid_standing_env.yaml"
ENGINE = "data/engines/newton_1_7_engine.yaml"
AGENTS = {
    "task_only": "data/agents/smp_getup_humanoid_standing_task_only_agent.yaml",
    "frozen_smp": "data/agents/smp_getup_humanoid_standing_agent.yaml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_envs", type=int, choices=(2048, 4096), default=4096)
    parser.add_argument("--max_samples", type=int, default=10_000_000)
    args = parser.parse_args()
    if args.max_samples <= 0:
        parser.error("--max_samples must be positive.")
    os.chdir(ROOT)
    evidence = [preflight(ENV, agent, ENGINE)[3] for agent in AGENTS.values()]
    if evidence[0]["prior_model_sha256"] != evidence[1]["prior_model_sha256"]:
        raise ValueError("Both controls must use the same frozen prior.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "started": datetime.now().astimezone().isoformat(),
              "prior": evidence[0], "num_envs": args.num_envs,
              "requested_samples_per_arm": args.max_samples, "arms": {}}
    def save():
        write_json(out / "summary.json", result)
    def run(command, log):
        with log.open("w", encoding="utf8") as stream:
            subprocess.run([sys.executable, "-B", "-u", *command], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    save()
    try:
        for name, agent in AGENTS.items():
            folder = out / name
            run(["tools/getup/train_getup.py", "--stage", "pilot", "--num_envs", str(args.num_envs),
                 "--max_samples", str(args.max_samples), "--env_config", ENV, "--agent_config", agent,
                 "--engine_config", ENGINE, "--out_dir", str(folder)], out / (name + "_train.txt"))
            result["arms"][name] = {}
            save()
            for label, init in (("standing", "standing"), ("late_reference", "reference"), ("fallen_starts", "starts")):
                dest = folder / ("eval_" + label)
                command = ["tools/getup/evaluate_getup.py", "--model_file", str(folder / "model.pt"),
                           "--env_config", ENV, "--agent_config", agent, "--engine_config", ENGINE,
                           "--init", init, "--episodes", "100", "--seed", "123", "--out_dir", str(dest)]
                if init == "reference":
                    command += ["--phase_range", "0.9", "1.0"]
                run(command, out / (name + "_" + label + ".txt"))
                result["arms"][name][label] = json.loads((dest / "evaluation.json").read_text())["overall"]
                save()
            print(name, result["arms"][name], flush=True)
        result["status"] = "complete"
    except BaseException as exc:
        result["status"] = "failed"
        result["error"] = repr(exc)
        raise
    finally:
        result["ended"] = datetime.now().astimezone().isoformat()
        save()


if __name__ == "__main__":
    main()
