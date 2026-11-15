"""Bounded read-only reset, reward and observation-normalizer ablations.

Runs no optimizer steps or normalizer updates. Crossed checkpoints exist only
in memory; none is exported as a policy. The source files remain untouched.
"""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'mimickit'))
from tools.getup.common import preflight, provenance, sha256, write_json
from tools.getup.near_standing_curriculum import AGENT, ENGINE, SOURCE, STAGES
from tools.getup.diagnose_getup import state_fingerprint
from envs.task_getup_env import TaskGetupEnv
from learning import agent_builder
from learning.base_agent import AgentMode
from util import mp_util, util


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--candidate', default='output/humanoid_getup_near_curriculum_r02/stage1/train/model.pt')
    parser.add_argument('--reference_probability', type=float, default=0.1,
                        help='Historical r02 used 0.1; keep diagnosis reproducible after configuration changes.')
    args = parser.parse_args()
    if not 0 < args.reference_probability < 1:
        parser.error('reference_probability must lie strictly between 0 and 1')
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    cfg, _, engine, _ = preflight(STAGES[0], AGENT, ENGINE)
    cfg['getup_near_reference_probability'] = args.reference_probability
    hashes = {p: sha256(p) for p in (args.source, args.candidate)}
    states = {k: torch.load(p, map_location='cuda:0') for k,p in [('source',args.source),('candidate',args.candidate)]}
    result = provenance({'sources': hashes, 'reference_probability': args.reference_probability,
                         'evaluation_steps': 600, 'evaluations': {}, 'signals': {}})
    mp_util.init(0,1,'cuda:0',0)
    env = agent = None
    try:
        util.set_rand_seed(456)
        env = TaskGetupEnv(cfg, engine, 100, 'cuda:0', False)
        agent = agent_builder.build_agent(AGENT, env, 'cuda:0')
        char = env._get_char_id()
        with torch.no_grad():
            # Same seeds, physical starts and deterministic actions across arms.
            for weights, norm in [('source','source'),('candidate','candidate'),('candidate','source'),('source','candidate')]:
                state = {k:(states[norm][k] if k.startswith('_obs_norm.') else v)
                         for k,v in states[weights].items()}
                agent.load_state_dict(state)
                agent.eval()
                agent.set_mode(AgentMode.TEST)
                util.set_rand_seed(456)
                obs, info = env.reset()
                before = state_fingerprint(agent)
                initial = {'root_z': float(env._engine.get_root_pos(char)[:,2].mean()),
                    'head_z': float(env._engine.get_body_pos(char)[:,env._head_id,2].mean()),
                    'root_speed': float(env._engine.get_root_vel(char).norm(dim=-1).mean()),
                    'dof_speed': float(env._engine.get_dof_vel(char).abs().mean()),
                    'joint_neutral_error': float((env._engine.get_dof_pos(char)-env._init_dof_pos).abs().mean()),
                    'obs_clip_fraction': float((agent._obs_norm.normalize(obs).abs()>=10).float().mean())}
                for _ in range(600):
                    action, _ = agent._decide_action(obs, info)
                    obs, _, done, info = env.step(action)
                    if bool((done == 1).any()):
                        raise RuntimeError('Unexpected early termination in ablation')
                assert state_fingerprint(agent) == before
                label = weights+'_weights_'+norm+'_obsnorm'
                result['evaluations'][label] = {'initial': initial,
                    'success': float(env._ever_success.float().mean()),
                    'end_stable': float((env._hold>=env._hold_target-1e-5).float().mean()),
                    'best_hold': float(env._best_hold.mean())}
                print('EVAL',label,json.dumps(result['evaluations'][label]),flush=True)

            # Independent short rollouts in the actual 90/10 training mixture.
            # Capture signal groups before any learning or statistics update.
            for label,state in states.items():
                agent.load_state_dict(state)
                agent.eval()
                agent.set_mode(AgentMode.TRAIN)
                util.set_rand_seed(123)
                agent._curr_obs, agent._curr_info = env.reset()
                agent._init_train()
                mask = env._init_static.clone()
                before = state_fingerprint(agent)
                initial = {
                    'root_z':env._engine.get_root_pos(char)[:,2].clone(),
                    'head_z':env._engine.get_body_pos(char)[:,env._head_id,2].clone(),
                    'root_speed':env._engine.get_root_vel(char).norm(dim=-1).clone(),
                    'dof_speed':env._engine.get_dof_vel(char).abs().mean(-1).clone(),
                    'joint_neutral_error':(env._engine.get_dof_pos(char)-env._init_dof_pos).abs().mean(-1).clone()}
                agent._rollout_train(32)
                task = agent._exp_buffer.get_data('reward').clone()
                agent._build_train_data()
                buf=agent._exp_buffer
                mixed=buf.get_data('reward')
                smp=(mixed-agent._task_reward_weight*task)/agent._smp_reward_weight
                values = {'task':task,'smp':smp,'mixed':mixed,'adv':buf.get_data('adv'),
                          'target':buf.get_data('tar_val'),
                          'obs_clip_fraction':(agent._obs_norm.normalize(buf.get_data('obs')).abs()>=10).float().mean(-1)}
                groups={}
                for name,m in [('standing',mask),('reference',~mask)]:
                    if not m.any():
                        raise RuntimeError('No samples in '+name)
                    groups[name]={'n':int(m.sum()), 'initial':{k:float(v[m].mean()) for k,v in initial.items()},
                                  **{k:float(v[:,m].mean()) for k,v in values.items()},
                                  'positive_adv_fraction':float((values['adv'][:,m]>0).float().mean())}
                assert state_fingerprint(agent) == before
                result['signals'][label]=groups
                print('SIGNAL',label,json.dumps(groups),flush=True)
        assert all(sha256(p)==h for p,h in hashes.items())
        result['source_files_unchanged']=True
        write_json(out/'probe.json',result)
    finally:
        if agent is not None:
            agent.close()
        if env is not None:
            env.close()
        mp_util.shutdown()


if __name__ == '__main__':
    main()
