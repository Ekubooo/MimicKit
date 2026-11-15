"""Bounded optimizer diagnostics; source checkpoint is never overwritten."""
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'mimickit'))
from tools.getup.common import preflight, provenance, sha256, write_json
from envs.task_getup_env import TaskGetupEnv
from learning import agent_builder
from util import mp_util, util


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--agent_config', default='data/agents/smp_getup_humanoid_standing_neutral_agent.yaml')
    parser.add_argument('--model_file', default='output/humanoid_getup_standing_neutral_pair_r01/frozen_smp/model.pt')
    parser.add_argument('--updates', type=int, default=2, choices=range(1, 6))
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    env_cfg, agent_cfg, engine_cfg, prior = preflight('data/envs/smp_getup_humanoid_near_stage1_env.yaml',
                args.agent_config, 'data/engines/newton_1_7_engine.yaml')
    source_hash = sha256(args.model_file)
    mp_util.init(0,1,'cuda:0',0)
    util.set_rand_seed(123)
    env = TaskGetupEnv(env_cfg, engine_cfg, 4096, 'cuda:0', False)
    agent = None
    trace, batches = [], []
    result = provenance({'source_sha256': source_hash, 'prior':prior, 'status':'running'})
    try:
        agent = agent_builder.build_agent(args.agent_config, env, 'cuda:0')
        agent.load(args.model_file)
        original_loss = agent._compute_critic_loss
        def critic_loss(batch):
            with torch.no_grad():
                obs = agent._obs_norm.normalize(batch['obs'])
                pred = agent._model.eval_critic(obs)
                batches.append({'iteration':agent._iter, 'obs_max':float(obs.abs().max()),
                    'obs_clip_fraction':float((obs.abs()>=10).float().mean()),
                    'target_max':float(batch['tar_val'].abs().max()), 'prediction_max':float(pred.abs().max()),
                    'obs_finite':bool(torch.isfinite(obs).all()),
                    'target_finite':bool(torch.isfinite(batch['tar_val']).all())})
            return original_loss(batch)
        agent._compute_critic_loss = critic_loss
        original_update = agent._update_critic
        def update_critic(batch_size, steps):
            # Capture the most recent rollout, optimizer momentum and sample
            # permutation before its critic updates, for controlled replay.
            if agent._iter >= 1:
                torch.save({'agent':agent.state_dict(),
                    'optimizer':agent._critic_optimizer._optimizer.state_dict(),
                    'obs':agent._exp_buffer.get_data_flat('obs'),
                    'tar_val':agent._exp_buffer.get_data_flat('tar_val'),
                    'sample_order':agent._exp_buffer._sample_buf,
                    'sample_head':agent._exp_buffer._sample_buf_head,
                    'batch_size':batch_size,'steps':steps},out/'critic_replay.pt')
            return original_update(batch_size,steps)
        agent._update_critic = update_critic
        for name, opt in [('critic',agent._critic_optimizer),('actor',agent._actor_optimizer)]:
            original = opt.step
            def checked(loss, opt=opt, name=name, original=original):
                row={'kind':name,'iteration':agent._iter,'step':opt.get_steps(),'loss':float(loss.detach())}
                trace.append(row)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{name}: nonfinite loss before optimizer step')
                original(loss)
                grads=[p.grad for p in opt._param_list if p.grad is not None]
                row['grad_norm']=float(torch.linalg.vector_norm(torch.stack([g.double().norm() for g in grads])))
                row['parameter_max']=max(float(p.detach().abs().max()) for p in opt._param_list)
                row['parameters_finite']=all(bool(torch.isfinite(p).all()) for p in opt._param_list)
                if not row['parameters_finite']:
                    raise FloatingPointError(f'{name}: nonfinite parameters after optimizer step')
            opt.step=checked
        agent.train_model(args.updates*4096*32,str(out),False,'txt')
        result['status']='complete'
    except FloatingPointError as exc:
        result.update(status='nonfinite_detected',error=str(exc))
    except BaseException as exc:
        result.update(status='failed',error=repr(exc))
        raise
    finally:
        result.update(trace=trace,critic_batches=batches,source_unchanged=sha256(args.model_file)==source_hash)
        write_json(out/'probe.json',result)
        print('PROBE_RESULT',result['status'],result.get('error'),flush=True)
        if agent is not None:
            agent.close()
        env.close()
        mp_util.shutdown()


if __name__ == '__main__':
    main()
