"""Causal-probe event timing: interruption, slot isolation, and a bounded single pulse."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.g1_dodgeball.control_probe import ReleaseSchedule


class ReleaseScheduleTest(unittest.TestCase):
    def test_requires_consecutive_pressure_after_launch_and_active_slot(self):
        timer = ReleaseSchedule((2, 2), "cpu", 0.1)
        q, request, low = torch.zeros(2, 2), -torch.ones(2, 2), torch.zeros(2)
        active = torch.tensor([True, False])
        launched = torch.tensor([False, True])
        for step in range(8):
            trigger, pulse = timer.update(q, request, low, active, launched, step)
            self.assertFalse(bool(trigger.any() | pulse.any()))
        launched[0] = True
        for step in range(8, 12):
            timer.update(q, request, low, active, launched, step)
        request[0, 0] = 0.2  # first knee loses consecutive-pressure history
        trigger, pulse = timer.update(q, request, low, active, launched, 12)
        self.assertEqual(trigger.tolist(), [[False, True], [False, False]])
        self.assertEqual(pulse.tolist(), [[False, True], [False, False]])

    def test_pulse_continues_after_recovery_and_never_retriggers(self):
        timer = ReleaseSchedule((1, 1), "cpu", 0.1)
        low, active = torch.zeros(1), torch.ones(1, dtype=torch.bool)
        q, request = torch.zeros(1, 1), -torch.ones(1, 1)
        pulses, triggers = [], []
        for step in range(30):
            if step == 5:
                q.fill_(0.6)
                request.fill_(0.6)
            if step == 15:
                q.zero_()
                request.fill_(-1)
            trigger, pulse = timer.update(q, request, low, active, active, step)
            if trigger.item():
                triggers.append(step)
            if pulse.item():
                pulses.append(step)
        self.assertEqual(triggers, [4])
        self.assertEqual(pulses, list(range(4, 14)))
        _, pulse = timer.update(q, request, low, ~active, active, 31)
        self.assertFalse(pulse.item())


if __name__ == "__main__":
    unittest.main()
