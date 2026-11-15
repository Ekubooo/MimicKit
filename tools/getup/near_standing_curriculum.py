"""Two bounded, gated near-standing stages; never starts full Getup training."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.getup.common import preflight, provenance, read_yaml, sha256, write_json

ENGINE = 'data/engines/newton_1_7_engine.yaml'
CLEAN_ENV = 'data/envs/smp_getup_humanoid_standing_neutral_env.yaml'
AGENT = 'data/agents/smp_getup_humanoid_near_agent.yaml'
SOURCE = 'output/humanoid_getup_standing_neutral_pair_r01/frozen_smp/model.pt'
STAGES = [f'data/envs/smp_getup_humanoid_near_stage{i}_env.yaml' for i in (1, 2)]


def advancement_reasons(before, after):
    """Engineering gates on separately labelled tests, not statistical claims."""
    reasons = []
    if any(r['failures'] for r in after.values()):
        reasons.append('evaluation_failure')
    if after['clean']['end_stable_rate'] < .95:
        reasons.append('clean_standing_regressed')
    if after['perturbed']['end_stable_rate'] < .8:
        reasons.append('perturbed_recovery_below_80pct')
    target = min(.95, max(.25, before['reference']['end_stable_rate'] + .1))
    if after['reference']['end_stable_rate'] + 1e-9 < target:
        reasons.append('reference_recovery_insufficient_improvement')
    return reasons


def verify_source(model, prior, agent):
    model = Path(model)
    manifest = json.loads((model.parent / 'run_manifest.json').read_text(encoding='utf8'))
    if manifest['prior_model_sha256'] != prior['prior_model_sha256']:
        raise ValueError('Source checkpoint provenance has a different frozen prior.')
    source_agent = read_yaml(model.parent / 'agent_config.yaml')
    # Loading a checkpoint restores the fixed std as well as weights. Reject
    # a config that would claim a different exploration distribution.
    for key in ('actor_std_type', 'action_std'):
        if source_agent['model'][key] != agent['model'][key]:
            raise ValueError('Warm-start action distribution mismatch: ' + key)
    return sha256(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model_file', default=SOURCE)
    parser.add_argument('--agent_config', default=AGENT)
    parser.add_argument('--num_envs', type=int, choices=(2048, 4096), default=4096)
    parser.add_argument('--max_samples_per_stage', type=int, default=4_194_304)
    parser.add_argument('--max_stages', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if not 0 < args.max_samples_per_stage <= 4_194_304:
        parser.error('Each stage is bounded to at most 4,194,304 requested samples.')
    os.chdir(ROOT)
    evidence = [preflight(env, args.agent_config, ENGINE) for env in STAGES]
    agent, prior = evidence[0][1], evidence[0][3]
    if agent['task_reward_weight'] != .5 or agent['smp_reward_weight'] != .5:
        raise ValueError('This curriculum preserves the existing frozen-SMP reward balance.')
    if not all(agent[name+'_optimizer'].get('check_finite', False) for name in ('actor','critic')):
        raise ValueError('Near-standing curriculum requires immediate finite optimizer checks.')
    source_hash = verify_source(args.model_file, prior, agent)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    summary = provenance({'status': 'running', 'started': datetime.now().astimezone().isoformat(),
        'source': args.model_file, 'source_sha256': source_hash, 'prior': prior,
        'num_envs': args.num_envs, 'requested_samples_per_stage': args.max_samples_per_stage,
        'max_stages': args.max_stages, 'stages': {}})
    def save():
        write_json(out / 'summary.json', summary)
    def run(command, log):
        with log.open('w', encoding='utf8') as stream:
            subprocess.run([sys.executable, '-B', '-u', *command], stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
    def evaluate(model, env, folder, include_fallen=False):
        folder.mkdir()
        specs = [('clean', CLEAN_ENV, 'standing'), ('perturbed', env, 'standing'),
                 ('reference', env, 'reference')]
        if include_fallen:
            specs.append(('fallen', env, 'starts'))
        results = {}
        for label, config, init in specs:
            dest = folder / label
            run(['tools/getup/evaluate_getup.py', '--model_file', str(model),
                 '--env_config', config, '--agent_config', args.agent_config, '--engine_config', ENGINE,
                 '--init', init, '--episodes', '100', '--seed', '456', '--out_dir', str(dest)], folder / (label+'.txt'))
            results[label] = json.loads((dest / 'evaluation.json').read_text())['overall']
        return results
    save()
    try:
        source = args.model_file
        for index, env in enumerate(STAGES[:args.max_stages], 1):
            key = f'stage{index}'
            folder = out / key
            folder.mkdir()
            record = {'source': str(source), 'source_sha256': verify_source(source, prior, agent),
                      'env_config': read_yaml(env), 'status': 'baseline'}
            summary['stages'][key] = record
            save()
            before = evaluate(source, env, folder / 'before')
            record['before'] = before
            if any(v['failures'] for v in before.values()) or before['clean']['end_stable_rate'] < .95:
                record.update(status='stopped', stop_reasons=['baseline_failed'])
                break
            record['status'] = 'training'
            save()
            train = folder / 'train'
            run(['tools/getup/train_getup.py', '--stage', 'pilot', '--model_file', str(source),
                 '--num_envs', str(args.num_envs), '--max_samples', str(args.max_samples_per_stage),
                 '--env_config', env, '--agent_config', args.agent_config, '--engine_config', ENGINE,
                 '--out_dir', str(train)], folder / 'train.txt')
            lines = (train / 'log.txt').read_text().splitlines()
            if len(lines) < 2 or not all(math.isfinite(float(v)) for line in lines[1:] for v in line.split()):
                raise RuntimeError('Missing or nonfinite training metrics; stopping curriculum.')
            source = train / 'model.pt'
            after = evaluate(source, env, folder / 'after', include_fallen=True)
            reasons = advancement_reasons(before, after)
            record.update(status='complete', after=after, advancement_passed=not reasons, stop_reasons=reasons)
            save()
            if reasons:
                break
        summary['status'] = 'complete'
    except BaseException as exc:
        summary.update(status='failed', error=repr(exc))
        raise
    finally:
        summary['ended'] = datetime.now().astimezone().isoformat()
        save()


if __name__ == '__main__':
    main()
