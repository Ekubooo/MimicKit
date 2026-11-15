"""Compare critic updates on identical captured data, weights and momentum."""
import argparse
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import gymnasium.spaces as spaces
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'mimickit'))
from learning.smp_model import SMPModel
from tools.getup.common import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True)
    parser.add_argument('--out_file', required=True)
    args=parser.parse_args()
    if Path(args.out_file).exists():
        raise FileExistsError(args.out_file)
    capture=torch.load(args.capture,map_location='cuda:0')
    state=capture['agent']
    cfg=yaml.safe_load((ROOT/'data/agents/smp_getup_humanoid_standing_neutral_agent.yaml').read_text())
    env=SimpleNamespace(get_obs_space=lambda: spaces.Box(-np.inf,np.inf,shape=tuple(state['_obs_norm._mean'].shape)),
                        get_action_space=lambda: spaces.Box(-1,1,shape=tuple(state['_a_norm._mean'].shape)))
    model=SMPModel(cfg['model'],env).to('cuda:0')
    norm=torch.clamp((capture['obs']-state['_obs_norm._mean'])/state['_obs_norm._std'],-10.,10.)
    target=capture['tar_val']
    order=capture['sample_order'];size=capture['batch_size']
    result={'obs_finite':bool(torch.isfinite(norm).all()),'obs_max':float(norm.abs().max()),
            'target_finite':bool(torch.isfinite(target).all()),'target_max':float(target.abs().max()),
            'sampling':'same captured permutation, cycled without reshuffling for all arms','arms':{}}
    for name,clip,lr in [('original',0,None),('clip10',10,None),('lr1e-5',0,1e-5)]:
        model.load_state_dict({k[len('_model.'):]:v for k,v in state.items() if k.startswith('_model.')})
        optimizer=torch.optim.SGD(model.get_critic_params(),lr=1e-4,momentum=.9)
        optimizer.load_state_dict(copy.deepcopy(capture['optimizer']))
        if lr is not None:
            for group in optimizer.param_groups:
                group['lr']=lr
        rows=[]
        for i in range(capture['steps']):
            indices=order[(torch.arange(size,device='cuda:0')+capture['sample_head']+i*size)%len(order)]
            prediction=model.eval_critic(norm[indices]).squeeze(-1)
            loss=(target[indices]-prediction).square().mean()
            row={'step':i,'loss':float(loss.detach()),'prediction_max':float(prediction.detach().abs().max())}
            rows.append(row)
            if not torch.isfinite(loss):
                break
            optimizer.zero_grad()
            loss.backward()
            params=model.get_critic_params()
            row['grad_norm']=float(torch.stack([p.grad.double().norm() for p in params]).norm())
            if not all(bool(torch.isfinite(p.grad).all()) for p in params):
                break
            if clip:
                torch.nn.utils.clip_grad_norm_(params,clip,error_if_nonfinite=True)
            optimizer.step()
        result['arms'][name]=rows
        print('REPLAY',name,'STEPS',len(rows),'LAST',rows[-1],flush=True)
    write_json(args.out_file,result)


if __name__ == '__main__':
    main()
