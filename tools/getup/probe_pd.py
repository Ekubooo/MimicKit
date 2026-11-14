"""Read-only, short matched controls for Getup PD targets and exploration."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mimickit'))

from tools.getup.common import write_json, provenance
from tools.getup.diagnose_getup import state_fingerprint
from envs.task_getup_env import TaskGetupEnv
from learning import agent_builder
from learning.base_agent import AgentMode
from util import mp_util, util


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--agent_config', default='data/agents/smp_getup_humanoid_standing_agent.yaml')
    parser.add_argument('--model_files', nargs='*', default=[])
    parser.add_argument('--init_noise', type=float, default=0.0)
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    mp_util.init(0, 1, 'cuda:0', 0)
    util.set_rand_seed(123)
    cfg = yaml.safe_load((ROOT / 'data/envs/smp_getup_humanoid_standing_env.yaml').read_text())
    cfg['getup_standing_joint_noise'] = args.init_noise
    engine = yaml.safe_load((ROOT / 'data/engines/newton_1_7_engine.yaml').read_text())
    labels = ['neutral', 'action_midpoint', 'neutral_noise_005', 'neutral_noise_00125',
              'neutral_noise_0003125', 'neutral_noise_000125', 'random_mode', 'random_sample',
              'centered_random_mode', 'centered_small_random_mode'] + [f'policy_{i}' for i in range(len(args.model_files))]
    width = 8
    env = TaskGetupEnv(cfg, engine, len(labels) * width, 'cuda:0', False)
    agents = []
    try:
        for model in [None] + args.model_files:
            agent = agent_builder.build_agent(args.agent_config, env, 'cuda:0')
            if model:
                agent.load(model)
            agent.eval()
            agent.set_mode(AgentMode.TEST)
            agents.append(agent)
        before = [state_fingerprint(a) for a in agents]
        obs, info = env.reset()
        # Match the eight perturbed initial poses across controls, rather than
        # confounding a noise/control change with a different starting pose.
        for k in range(len(labels)):
            util.set_rand_seed(999)
            obs, info = env.reset(torch.arange(k*width, (k+1)*width, device='cuda:0'))
        dofs = env._engine.get_dof_pos(env._get_char_id())
        for k in range(1, len(labels)):
            torch.testing.assert_close(dofs[:width], dofs[k*width:(k+1)*width])
        util.set_rand_seed(123)
        neutral = env._init_dof_pos
        space = env.get_action_space()
        mid = torch.as_tensor((space.low + space.high) / 2, device='cuda:0')
        scale = torch.as_tensor((space.high - space.low) / 2, device='cuda:0')
        rows = []
        mapping = {'neutral': neutral.tolist(), 'action_midpoint': mid.tolist(),
                   'physical_std_at_005': (.05 * scale).tolist(), 'joints': []}
        for j in range(1, env._kin_char_model.get_num_joints()):
            joint = env._kin_char_model.get_joint(j)
            mapping['joints'].append({'index': j, 'name': joint.name, 'neutral': joint.get_joint_dof(neutral).tolist(),
                                     'midpoint': joint.get_joint_dof(mid).tolist()})
        fall = torch.full((env.get_num_envs(),), -1., device='cuda:0')
        sums = torch.zeros_like(fall)
        target_error = torch.zeros_like(fall)
        first = None
        for step in range(300):
            noise = torch.randn(width, neutral.numel(), device='cuda:0')
            targets = [neutral.expand(width, -1), mid.expand(width, -1),
                       neutral + .05 * scale * noise, neutral + .0125 * scale * noise,
                       neutral + .003125 * scale * noise, neutral + .00125 * scale * noise]
            for index, a in enumerate(agents):
                mode = a._a_norm.unnormalize(a._model.eval_actor(a._obs_norm.normalize(obs)).mode)
                if index == 0:
                    targets.extend([mode[6*width:7*width], mode[7*width:8*width] + .05 * scale * noise,
                                    neutral + mode[8*width:9*width] - mid,
                                    neutral + .1 * (mode[9*width:10*width] - mid)])
                else:
                    k = 9 + index
                    targets.append(mode[k*width:(k+1)*width])
            action = torch.cat(targets)
            if first is None:
                first = action.clone()
            target_error += (action - neutral).abs().mean(dim=1)
            obs, reward, done, info = env.step(action)
            # Newton's command buffer must receive the environment's clipped PD targets verbatim.
            torch.testing.assert_close(env._engine._controls.target_pos[env._get_char_id()], env._clip_action(action))
            head = env._engine.get_body_pos(env._get_char_id())[:, env._head_id, 2]
            fallen = (head < .6) & (fall < 0)
            fall[fallen] = (step + 1) / 30
            sums += reward
            if bool((done != 0).any()):
                raise RuntimeError('Unexpected termination in 10-second control probe.')
        for k, label in enumerate(labels):
            sl = slice(k*width, (k+1)*width)
            rows.append({'control': label, 'episodes': width,
                         'successes': int(env._ever_success[sl].sum()),
                         'end_stable': int((env._hold[sl] >= 5-1e-5).sum()),
                         'mean_best_hold': float(env._best_hold[sl].mean()),
                         'mean_final_head': float(head[sl].mean()),
                         'fall_times': fall[sl].tolist(),
                         'mean_task_reward': float(sums[sl].mean()/300),
                         'first_target': first[sl].mean(0).tolist(),
                         'mean_target_error_rad': float(target_error[sl].mean()/300)})
        assert before == [state_fingerprint(a) for a in agents], 'Probe mutated agent state.'
        write_json(out / 'summary.json', provenance({'init_noise': args.init_noise,
                   'models': args.model_files, 'mapping': mapping, 'controls': rows,
                   'unchanged_agents': True, 'command_buffer_verified': True}))
        for r in rows:
            print('RESULT', r['control'], r['successes'], round(r['mean_best_hold'],3),
                  round(r['mean_final_head'],3), round(r['mean_target_error_rad'],3), flush=True)
    finally:
        for a in agents:
            a.close()
        env.close()
        mp_util.shutdown()


if __name__ == '__main__':
    main()
