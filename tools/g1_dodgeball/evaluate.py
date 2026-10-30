"""Exactly one first episode per parallel G1 dodgeball environment; no learning."""
import argparse
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import torch
import yaml
from tools.g1_dodgeball.workflow import ENV, AGENT, ENGINE, PRIOR_DIR, audit_assets, check_prior
from tools.getup.common import read_yaml, provenance, sha256, write_json
from tools.getup.evaluate_getup import Tee
from tools.getup.diagnose_getup import score_windows


@torch.no_grad()
def run(args, out):
    from envs.task_dodgeball_env import TaskDodgeballEnv, compute_dodgeball_fail_flags
    from envs.base_env import DoneFlags
    from learning import agent_builder
    from learning.base_agent import AgentMode
    from util import mp_util, resource_util, util
    audit = audit_assets(args.device)
    check_prior(args.prior_dir, audit)
    folder = Path(args.model_file).parent
    if folder.name == "int_models":
        folder = folder.parent
    def saved_config(name, fallback):
        p = folder / name
        return read_yaml(p if p.is_file() else fallback)
    env_cfg = saved_config("env_config.yaml", ENV)
    engine_cfg = saved_config("engine_config.yaml", ENGINE)
    agent_cfg = saved_config("agent_config.yaml", AGENT)
    if env_cfg != read_yaml(ENV) or engine_cfg != read_yaml(ENGINE):
        raise ValueError("Saved policy environment/engine differs from this preset; inspect before evaluating.")
    env_cfg["rand_reset"] = args.init != "starts"
    agent_cfg.update(enable_gsi=args.init == "gsi",
                     smp_prior_cfg=str(Path(args.prior_dir) / "diffusion_config.yaml"),
                     smp_prior_model=str(Path(args.prior_dir) / "model.pt"))
    for name, cfg in (("env_config.yaml", env_cfg), ("engine_config.yaml", engine_cfg),
                      ("agent_config.yaml", agent_cfg)):
        (out / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    util.set_rand_seed(args.seed)
    env = agent = None
    started = time.monotonic()
    mp_util.init(0, 1, args.device, 0)
    try:
        env = resource_util.construct(TaskDodgeballEnv, env_config=env_cfg, engine_config=engine_cfg,
                                      num_envs=args.episodes, device=args.device, visualize=False)
        agent = agent_builder.build_agent(str(out / "agent_config.yaml"), env, args.device)
        agent.load(args.model_file)
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        obs, info = env.reset()
        n, char, dt = args.episodes, env._get_char_id(), env._engine.get_timestep()
        seen = torch.zeros(n, dtype=torch.bool, device=args.device)
        returns = torch.zeros(n, device=args.device)
        launches = torch.zeros(n, dtype=torch.long, device=args.device)
        hit_seen, fall_seen = seen.clone(), seen.clone()
        windows, slots, rows = [], [], []
        for step in range(math.ceil(env_cfg["episode_length"] / dt) + 3):
            active = ~seen
            trigger_before = env._proj_trigger_times.clone()
            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            if not bool(torch.isfinite(obs).all() & torch.isfinite(info["disc_obs"]).all() & torch.isfinite(reward).all()):
                raise RuntimeError("Non-finite policy/prior observations or task rewards.")
            returns += reward * active
            launches += (env._proj_trigger_times != trigger_before).sum(dim=1) * active
            pos, vel = env._get_proj_states()
            hit = compute_dodgeball_fail_flags(env._engine.get_body_pos(char), pos, vel, env._prev_proj_vel,
                    env._get_proj_contact_force(), env._hit_dist, env._hit_force_threshold,
                    env._hit_delta_v_threshold, env._engine.get_gravity()[2], dt)
            ground = env._engine.get_ground_contact_forces(char).clone()
            ground[:, env._contact_body_ids, :] = 0
            fall = ground.abs().amax(dim=-1).amax(dim=-1) > 0.1
            hit_seen |= hit & active
            fall_seen |= fall & active
            if step % 3 == 0:
                ids = active.nonzero(as_tuple=False).flatten()
                windows.append(info["disc_obs"][ids].cpu().clone())
                slots.append(ids.cpu())
            terminal = done != DoneFlags.NULL.value
            for i in (terminal & active).nonzero(as_tuple=False).flatten().tolist():
                rows.append({"env_id": i, "episode_seconds": float(env.get_env_time()[i]),
                             "done": DoneFlags(int(done[i])).name, "task_return": float(returns[i]),
                             "launches": int(launches[i]), "hit_detected": bool(hit_seen[i]),
                             "nonfoot_ground_contact_detected": bool(fall_seen[i])})
            seen |= terminal
            if bool(seen.all()):
                break
            obs, info = env.reset(terminal.nonzero(as_tuple=False).flatten())
        if len(rows) != n:
            raise RuntimeError("Incomplete first-episode evaluation; no completed summary emitted.")
        raw, _, smp = score_windows(agent, torch.cat(windows), args.seed + 1000, repeats=2)
        if not bool(torch.isfinite(raw).all() & torch.isfinite(smp).all()):
            raise RuntimeError("Non-finite SDS/reward diagnostic.")
        ids = torch.cat(slots)
        for row in rows:
            mask = ids == row["env_id"]
            row.update(smp_reward_mean=float(smp[mask].mean()), raw_sds_mean=float(raw[mask].mean()))
        refs = env._compute_disc_obs_demo(torch.arange(2, device=args.device).repeat_interleave(32),
            torch.linspace(0, 1, 32, device=args.device).repeat(2) * env._motion_lib.get_motion_length(
                torch.arange(2, device=args.device).repeat_interleave(32)))
        ref_raw, _, ref_reward = score_windows(agent, refs, args.seed + 2000, repeats=4)
        rows.sort(key=lambda r: r["env_id"])
        with (out / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = provenance({"policy": args.model_file, "policy_sha256": sha256(args.model_file),
            "prior_model_sha256": sha256(Path(args.prior_dir) / "model.pt"),
            "seed": args.seed, "initialization": args.init, "episodes": n, "parallel_envs": n,
            "mean_episode_seconds": sum(r["episode_seconds"] for r in rows) / n,
            "timeout_rate": sum(r["done"] == "TIME" for r in rows) / n,
            "failures": sum(r["done"] == "FAIL" for r in rows),
            "episodes_with_hit_detector": sum(r["hit_detected"] for r in rows),
            "episodes_with_nonfoot_ground_contact": sum(r["nonfoot_ground_contact_detected"] for r in rows),
            "mean_task_return": sum(r["task_return"] for r in rows) / n,
            "mean_projectile_launches": sum(r["launches"] for r in rows) / n,
            "mean_smp_reward": float(smp.mean()), "raw_sds_mean": float(raw.mean()),
            "mean_episode_smp_reward": sum(r["smp_reward_mean"] for r in rows) / n,
            "reference_raw_sds_mean": float(ref_raw.mean()), "reference_smp_reward_mean": float(ref_reward.mean()),
            "sds_normalizer_count": agent._sds_normalizer.get_count().cpu().tolist(),
            "motion_audit": audit, "elapsed_seconds": time.monotonic() - started,
            "notes": ["Exactly the first episode in each slot; terminal failures included.",
                      "GSI uses generated prior states; reference uses random walk/run phase; starts uses frame 0 velocities.",
                      "Timeout is survival under this task's hit/fall detectors, not certified dodge success.",
                      "Hit and ground-contact categories may overlap; launches are not confirmed encounters.",
                      "Task return is the unweighted environment reward sum, not combined PPO reward.",
                      "SMP diagnostics sample windows at 10 Hz with the saved frozen SDS normalizer."]})
        write_json(out / "evaluation.json", result)
        (out / "summary.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Completed. Send: " + str(out / "summary.txt"), flush=True)
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None, mp_util.shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prior_dir")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--init", choices=("gsi", "reference", "starts"), default="gsi")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if not Path(args.model_file).is_file():
        raise FileNotFoundError(args.model_file)
    if args.prior_dir is None:
        folder = Path(args.model_file).parent
        if folder.name == "int_models":
            folder = folder.parent
        path = folder / "agent_config.yaml"
        args.prior_dir = str(Path(read_yaml(path)["smp_prior_cfg"]).parent) if path.is_file() else PRIOR_DIR
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, out)
            except KeyboardInterrupt:
                print("Interrupted; no completed evaluation claimed.")
                raise SystemExit(130)
            except Exception:
                import traceback
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
