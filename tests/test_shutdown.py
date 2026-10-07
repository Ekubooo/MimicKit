"""Regression checks for viewer exits and interrupted subprocess cleanup."""
import os
from pathlib import Path
import runpy
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
import run


class WorkerShutdownTests(unittest.TestCase):
    def run_main(self, proc, error=None):
        args = Mock()
        args.parse_int.return_value = 6500
        args.parse_strings.return_value = ["cpu", "cpu"]
        with patch.object(run, "load_args", return_value=args), \
             patch.object(run.torch.multiprocessing, "set_start_method"), \
             patch.object(run.torch.multiprocessing, "Process", return_value=proc), \
             patch.object(run, "run", side_effect=error):
            run.main(["run.py"])

    def test_normal_completion_joins_without_termination(self):
        proc = Mock()
        proc.is_alive.return_value = False
        self.run_main(proc)
        proc.join.assert_any_call()
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()

    def test_viewer_exit_reaps_worker_and_preserves_success_code(self):
        proc = Mock()
        proc.is_alive.side_effect = [True, False]
        with self.assertRaises(SystemExit) as ctx:
            self.run_main(proc, SystemExit(0))
        self.assertEqual(ctx.exception.code, 0)
        proc.terminate.assert_called_once()
        proc.join.assert_called_once_with(timeout=5)

    def test_interrupt_kills_worker_if_termination_times_out(self):
        proc = Mock()
        proc.is_alive.side_effect = [True, True]
        with self.assertRaises(KeyboardInterrupt):
            self.run_main(proc, KeyboardInterrupt())
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        proc.join.assert_any_call(timeout=5)
        proc.join.assert_any_call()

    def test_worker_interrupt_has_no_exception_traceback(self):
        with patch.object(run, "run", side_effect=KeyboardInterrupt()), \
             self.assertRaises(SystemExit) as ctx:
            run._run_worker()
        self.assertEqual(ctx.exception.code, 130)

    def test_unexpected_error_is_preserved_after_cleanup(self):
        proc = Mock()
        proc.is_alive.side_effect = [True, False]
        with self.assertRaisesRegex(RuntimeError, "simulation failed"):
            self.run_main(proc, RuntimeError("simulation failed"))
        proc.terminate.assert_called_once()


class ViewerShutdownTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from engines.newton_engine import NewtonEngine
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        cls.engine_type = NewtonEngine

    def make_engine(self, running):
        engine = self.engine_type.__new__(self.engine_type)
        engine._viewer = Mock()
        engine._viewer.is_running.side_effect = running
        engine._sim_state = Mock()
        engine.get_sim_time = Mock(return_value=1.0)
        engine._draw_line_count = 10
        return engine

    def test_already_closed_skips_all_rendering(self):
        engine = self.make_engine([False])
        with patch("engines.engine.Engine.render") as base_render, \
             self.assertRaises(SystemExit) as ctx:
            engine.render()
        self.assertEqual(ctx.exception.code, 0)
        engine._viewer.end_frame.assert_not_called()
        engine._viewer.begin_frame.assert_not_called()
        engine._viewer.log_state.assert_not_called()
        base_render.assert_not_called()

    def test_close_during_events_skips_calls_to_closed_context(self):
        engine = self.make_engine([True, False])
        with patch("engines.engine.Engine.render") as base_render, \
             self.assertRaises(SystemExit) as ctx:
            engine.render()
        self.assertEqual(ctx.exception.code, 0)
        engine._viewer.end_frame.assert_called_once()
        engine._viewer.begin_frame.assert_not_called()
        engine._viewer.log_state.assert_not_called()
        base_render.assert_not_called()

    def test_open_viewer_renders_normally(self):
        engine = self.make_engine([True, True])
        with patch("engines.engine.Engine.render") as base_render:
            engine.render()
        engine._viewer.begin_frame.assert_called_once_with(1.0)
        engine._viewer.log_state.assert_called_once_with(engine._sim_state.raw_state)
        base_render.assert_called_once()
        self.assertEqual(engine._draw_line_count, 0)


@unittest.skipUnless((ROOT / "output/environment_audit/launch.py").exists(), "Local demo launcher not present")
class LauncherShutdownTests(unittest.TestCase):
    def launch(self, proc):
        previous_cwd = os.getcwd()
        try:
            with patch.object(sys, "argv", ["launch.py", "smp_steering_humanoid"]), \
                 patch("subprocess.Popen", return_value=proc):
                with self.assertRaises(SystemExit) as ctx:
                    runpy.run_path(str(ROOT / "output/environment_audit/launch.py"), run_name="__main__")
                return ctx.exception.code
        finally:
            os.chdir(previous_cwd)

    def test_child_exit_code_is_forwarded(self):
        proc = Mock()
        proc.wait.return_value = 7
        self.assertEqual(self.launch(proc), 7)
        proc.terminate.assert_not_called()

    def test_interrupt_allows_child_to_exit(self):
        proc = Mock()
        proc.wait.side_effect = [KeyboardInterrupt(), 130]
        self.assertEqual(self.launch(proc), 130)
        proc.wait.assert_any_call(timeout=2)
        proc.terminate.assert_not_called()

    def test_unresponsive_child_is_killed_and_reaped(self):
        proc = Mock()
        timeout = subprocess.TimeoutExpired("demo", 2)
        proc.wait.side_effect = [KeyboardInterrupt(), timeout, timeout, 1]
        self.assertEqual(self.launch(proc), 130)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        self.assertEqual(proc.wait.call_count, 4)


if __name__ == "__main__":
    unittest.main()
