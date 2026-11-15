"""Reject corrupt optimizer steps before they propagate to later PPO updates."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mimickit'))
from learning.mp_optimizer import MPOptimizer


class FiniteOptimizerTests(unittest.TestCase):
    def optimizer(self, parameter, **extra):
        return MPOptimizer({'type':'SGD','learning_rate':.1,'check_finite':True,
                            'label':'test critic',**extra},[parameter])

    def test_nonfinite_loss_does_not_change_parameter_or_momentum(self):
        p=torch.nn.Parameter(torch.tensor(2.));o=self.optimizer(p)
        with self.assertRaisesRegex(FloatingPointError,'loss before backward'):
            o.step(p*float('nan'))
        self.assertEqual(p.item(),2.)
        self.assertEqual(o.get_steps(),0)
        self.assertEqual(len(o._optimizer.state),0)

    def test_finite_loss_with_infinite_gradient_is_rejected(self):
        p=torch.nn.Parameter(torch.tensor(0.));o=self.optimizer(p,grad_clip=10.)
        with self.assertRaisesRegex(FloatingPointError,'gradient before step'):
            o.step(p.sqrt())
        self.assertEqual(p.item(),0.)
        self.assertEqual(o.get_steps(),0)
        self.assertEqual(len(o._optimizer.state),0)

    def test_large_finite_gradient_is_bounded_and_step_runs(self):
        p=torch.nn.Parameter(torch.tensor(2.));o=self.optimizer(p,grad_clip=10.)
        o.step(10000*p.square())
        self.assertAlmostEqual(p.item(),1.,places=5)
        self.assertEqual(o.get_steps(),1)

    def test_parameter_overflow_stops_before_next_step(self):
        p=torch.nn.Parameter(torch.tensor(-3e38));o=self.optimizer(p,learning_rate=3e38)
        with self.assertRaisesRegex(FloatingPointError,'parameter after step'):
            o.step(p)
        self.assertEqual(o.get_steps(),0)


if __name__ == '__main__':
    unittest.main()
