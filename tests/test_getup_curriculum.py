"""Safety gates and warm-start provenance for the bounded recovery curriculum."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml
from tools.getup.near_standing_curriculum import advancement_reasons, verify_source, validate_rollout_horizon
from tools.getup.common import sha256


class CurriculumContracts(unittest.TestCase):
    def test_evaluation_must_allow_an_actual_training_episode(self):
        env={'episode_length':20.0}
        agent={'iters_per_output':5,'steps_per_iter':32}
        engine={'control_freq':30}
        with self.assertRaisesRegex(ValueError, 'truncates every training episode'):
            validate_rollout_horizon(env,agent,engine)
        agent['iters_per_output']=20
        validate_rollout_horizon(env,agent,engine)

    def test_advance_requires_recovery_and_retained_standing(self):
        before = {'reference': {'end_stable_rate': .1}}
        after = {k: {'end_stable_rate': v, 'failures': 0}
                 for k,v in [('clean',1.),('perturbed',.8),('reference',.25),('fallen',0.)]}
        self.assertEqual(advancement_reasons(before, after), [])
        for label, value in [('clean',.94),('perturbed',.79),('reference',.24)]:
            bad = copy.deepcopy(after)
            bad[label]['end_stable_rate'] = value
            self.assertTrue(advancement_reasons(before,bad))
        after['fallen']['failures'] = 1
        self.assertIn('evaluation_failure',advancement_reasons(before,after))

    def test_existing_reference_skill_does_not_count_as_improvement(self):
        after = {k: {'end_stable_rate': 1., 'failures': 0} for k in ['clean','perturbed','reference']}
        after['reference']['end_stable_rate'] = .4
        self.assertTrue(advancement_reasons({'reference':{'end_stable_rate':.4}},after))
        after['reference']['end_stable_rate'] = .5
        self.assertEqual(advancement_reasons({'reference':{'end_stable_rate':.4}},after), [])

    def test_source_prior_and_saved_exploration_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); model=p/'model.pt'; model.write_bytes(b'checkpoint fixture')
            (p/'run_manifest.json').write_text(json.dumps({'prior_model_sha256':'same'}))
            agent={'model':{'actor_std_type':'FIXED','action_std':.003125}}
            (p/'agent_config.yaml').write_text(yaml.safe_dump(agent))
            self.assertEqual(verify_source(model,{'prior_model_sha256':'same'},agent),sha256(model))
            with self.assertRaises(ValueError):
                verify_source(model,{'prior_model_sha256':'different'},agent)
            agent['model']['action_std']=.05
            with self.assertRaises(ValueError):
                verify_source(model,{'prior_model_sha256':'same'},agent)


if __name__ == '__main__':
    unittest.main()
