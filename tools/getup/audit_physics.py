"""Small Getup physics probes without training or learned-policy assumptions.

PD reference playback is an open-loop feasibility probe, not a learned get-up
controller. Failure of playback alone does not prove a motion is infeasible.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

import torch
import yaml
from envs.task_getup_env import TaskGetupEnv, compute_getup_reward
from util import mp_util, resource_util, util
from tools.getup.common import provenance, write_json, preflight, AGENT_CONFIG, ENV_CONFIG


@torch.no_grad()
def audit(env, seconds, agent=None):
    device = env._device
    ids = torch.arange(2, device=device)
    lengths = env._motion_lib.get_motion_length(ids)
    result = {"reference": {}, "probes": {}}
    for mid in range(2):
        times = torch.linspace(0, lengths[mid], 128, device=device)
        mids = torch.full((128,), mid, device=device, dtype=torch.long)
        pos, rot, _, _, joints, _ = env._motion_lib.calc_motion_frame(mids, times)
        bodies, rotations = env._kin_char_model.forward_kinematics(pos, rot, joints)
        clearance = env._supports.minimum_height(bodies, rotations)
        dofs = env._kin_char_model.rot_to_dof(joints)
        violation = torch.maximum(env._dof_low - dofs, dofs - env._dof_high).clamp_min(0)
        result["reference"][str(mid)] = {
            "minimum_clearance_m": float(clearance.min()),
            "maximum_reset_lift_m": float((env._clearance - clearance).clamp_min(0).max()),
            "maximum_limit_violation_rad": float(violation.max()),
            "fraction_frames_outside_limits": float((violation > .01).any(-1).float().mean()),
        }
    original_reset = env._reset_ref_motion
    for kind in ("reference_playback", "terminal_pose_hold", "neutral_pose_hold"):
        times = torch.zeros(2, device=device) if kind == "reference_playback" else lengths
        env._sample_motion_times = lambda n: (ids.clone(), times.clone())
        def reset(selected):
            original_reset(selected)
            if kind == "neutral_pose_hold":
                env._ref_root_pos[selected] = env._init_root_pos
                env._ref_root_rot[selected] = env._init_root_rot
                env._ref_dof_pos[selected] = env._init_dof_pos
                env._ref_joint_rot[selected] = env._kin_char_model.dof_to_rot(env._ref_dof_pos[selected])
                p, r = env._kin_char_model.forward_kinematics(env._ref_root_pos[selected], env._ref_root_rot[selected], env._ref_joint_rot[selected])
                lift = (env._clearance - env._supports.minimum_height(p, r)).clamp_min(0)
                env._ref_root_pos[selected, 2] += lift
                p[..., 2] += lift[:, None]
                env._ref_body_pos[selected], env._ref_body_rot[selected] = p, r
            if kind != "reference_playback":
                env._ref_root_vel[selected] = 0
                env._ref_root_ang_vel[selected] = 0
                env._ref_dof_vel[selected] = 0
        env._reset_ref_motion = reset
        env.reset()
        target = env._ref_dof_pos.clone()
        head, reward, stable, windows = [], [], [], []
        finite = True
        for step in range(round(seconds / env._engine.get_timestep())):
            if kind == "reference_playback":
                t = torch.minimum(torch.full_like(lengths, (step + 1) * env._engine.get_timestep()), lengths)
                joints = env._motion_lib.calc_motion_frame(ids, t)[4]
                target = env._kin_char_model.rot_to_dof(joints)
            obs, rew, done, info = env.step(target)
            raw = env._engine.get_body_pos(env._get_char_id())
            finite = finite and bool(torch.isfinite(raw).all())
            head.append(raw[:, env._head_id, 2].clone())
            reward.append(rew.clone())
            stable.append(env._standing.clone())
            if agent is not None:
                windows.append(info["disc_obs"].detach().cpu().clone())
            if bool((done != 0).any()):
                break
        heights = torch.stack(head)
        result["probes"][kind] = {
            "finite": finite, "steps": len(head),
            "final_head_height": heights[-1].tolist(), "max_head_height": heights.max(0).values.tolist(),
            "mean_task_reward": torch.stack(reward).mean(0).tolist(),
            "standing_fraction": torch.stack(stable).float().mean(0).tolist(),
            "best_hold_seconds": env._best_hold.tolist(), "success": env._ever_success.tolist(),
        }
        if agent is not None and len(windows) >= env._num_disc_obs_steps:
            from tools.getup.diagnose_getup import score_windows
            # Custom static resets do not represent a reference-motion history.
            # Discard the initial history; score only fully simulated windows.
            skip = env._num_disc_obs_steps - 1
            _, _, scores = score_windows(agent, torch.cat(windows[skip:]), 2123, 4)
            scores = scores.reshape(-1, 2)
            task = torch.stack(reward[skip:]).cpu()
            result["probes"][kind]["frozen_scores"] = {
                "smp_mean": scores.mean(0).tolist(),
                "task_mean": task.mean(0).tolist(),
                "paper_combined_mean": (.2 * task + .8 * scores).mean(0).tolist(),
                "balanced_combined_mean": (.5 * task + .5 * scores).mean(0).tolist(),
            }
    h = torch.tensor([.2, .59, .60, .8, 1.2], device=device)
    result["stationary_reward_table"] = {
        "head_height": h.tolist(),
        "deficit": compute_getup_reward(h, torch.zeros_like(h), 1.2, .6, .25, 6., 100., False).tolist(),
        "paper_literal": compute_getup_reward(h, torch.zeros_like(h), 1.2, .6, .25, 6., 100., True).tolist(),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine_config", default="data/engines/newton_1_7_engine.yaml")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--model_file", help="Optional old RL checkpoint, only for its frozen prior/normalizer.")
    parser.add_argument("--agent_config", default=AGENT_CONFIG)
    args = parser.parse_args()
    if not 1 <= args.seconds < 20:
        parser.error("Probe duration must be at least 1 and less than 20 seconds.")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    env = agent = None
    mp_util.init(0, 1, "cuda:0", 0)
    try:
        util.set_rand_seed(123)
        cfg = yaml.safe_load(Path(ENV_CONFIG).read_text())
        cfg["augmented_reset_probability"] = 0
        engine = yaml.safe_load(Path(args.engine_config).read_text())
        env = resource_util.construct(TaskGetupEnv, env_config=cfg, engine_config=engine,
                                     num_envs=2, device="cuda:0", visualize=False)
        extra = {}
        if args.model_file:
            from learning import agent_builder
            from learning.base_agent import AgentMode
            from tools.getup.diagnose_getup import state_fingerprint
            extra = preflight(ENV_CONFIG, args.agent_config, args.engine_config)[3]
            agent = agent_builder.build_agent(args.agent_config, env, "cuda:0")
            agent.load(args.model_file)
            agent.eval()
            agent.set_mode(AgentMode.TEST)
            before = state_fingerprint(agent)
        result = provenance({**extra, "engine_config": engine, **audit(env, args.seconds, agent)})
        if agent is not None:
            if before != state_fingerprint(agent):
                raise RuntimeError("Read-only probe changed agent/prior/normalizer state.")
            result["unchanged_agent_state_sha256"] = before
            result["scoring_checkpoint"] = args.model_file
        write_json(out / "summary.json", result)
        print(json.dumps(result, indent=2))
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None)
        mp_util.shutdown()


if __name__ == "__main__":
    main()
