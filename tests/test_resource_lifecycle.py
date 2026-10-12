"""Verify owned resources are released on success and all exit paths."""
import ast
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
import run
from util import display, mp_util, resource_util, video
from util.logger import Logger
from util.tb_logger import TBLogger
from util.wandb_logger import WandbLogger
from engines.engine import Engine


def backend_methods(filename, class_name, methods, base=Engine, **namespace):
    """Test real teardown methods without installing optional simulators."""
    tree = ast.parse((ROOT / "mimickit/engines" / filename).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    cls = ast.ClassDef(name=class_name, bases=[ast.Name(id="Base", ctx=ast.Load())],
                       keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {"Base": base, "resource_util": resource_util, **namespace}
    exec(compile(module, filename, "exec"), scope)
    return scope[class_name]


class CleanupTests(unittest.TestCase):
    def test_cleanup_attempts_every_resource_and_reports_failure(self):
        failed = Mock(side_effect=RuntimeError("close failed"))
        remaining = Mock()
        with self.assertRaisesRegex(RuntimeError, "close failed"):
            resource_util.cleanup(failed, remaining)
        remaining.assert_called_once()

    def test_cleanup_retains_original_exception(self):
        failed = Mock(side_effect=RuntimeError("close failed"))
        with self.assertRaisesRegex(ValueError, "original failure"):
            try:
                raise ValueError("original failure")
            finally:
                with self.assertWarnsRegex(RuntimeWarning, "close failed"):
                    resource_util.cleanup(failed)

    def test_failed_constructor_closes_partially_built_resource(self):
        closed = []
        class Partial:
            def __init__(self):
                self.handle = "allocated"
                raise ValueError("initialization failed")
            def close(self):
                closed.append(self.handle)
        with self.assertRaisesRegex(ValueError, "initialization failed"):
            resource_util.construct(Partial)
        self.assertEqual(closed, ["allocated"])

    def test_failed_constructor_preserves_error_if_close_also_fails(self):
        class Partial:
            def __init__(self):
                raise ValueError("initialization failed")
            def close(self):
                raise RuntimeError("cleanup failed")
        with self.assertWarns(RuntimeWarning), self.assertRaisesRegex(ValueError, "initialization failed"):
            resource_util.construct(Partial)


class RunBoundaryTests(unittest.TestCase):
    def exercise(self, mode="test", error=None, load_error=None, build_error=None):
        env = Mock()
        agent = Mock()
        agent.load.side_effect = load_error
        order = []
        env.close.side_effect = lambda: order.append("env")
        agent.close.side_effect = lambda: order.append("agent")
        args = run.load_args(["run.py", "--mode", mode, "--model_file", "probe.pt"])
        with patch.object(run.mp_util, "init"), patch.object(run, "set_rand_seed"), \
             patch.object(run, "set_np_formatting"), patch.object(run, "create_output_dir"), \
             patch.object(run, "save_config_files"), \
             patch.object(run, "build_env", return_value=env), \
             patch.object(run, "build_agent", return_value=agent, side_effect=build_error), \
             patch.object(run, "test", side_effect=error), patch.object(run, "train", side_effect=error), \
             patch.object(run.display, "close_virtual_display", side_effect=lambda: order.append("display")):
            try:
                run.run(0, 1, "cpu", 6500, args)
            finally:
                self.assertEqual(order, ["env", "display"] if build_error else ["env", "agent", "display"])
        return env, agent

    def test_natural_test_completion(self):
        self.exercise()

    def test_natural_train_completion(self):
        self.exercise(mode="train")

    def test_user_window_exit(self):
        with self.assertRaises(SystemExit) as ctx:
            self.exercise(error=SystemExit(0))
        self.assertEqual(ctx.exception.code, 0)

    def test_keyboard_interrupt(self):
        with self.assertRaises(KeyboardInterrupt):
            self.exercise(error=KeyboardInterrupt())

    def test_failed_model_load(self):
        with self.assertRaises(FileNotFoundError):
            self.exercise(load_error=FileNotFoundError("probe.pt"))

    def test_failed_agent_build(self):
        with self.assertRaisesRegex(RuntimeError, "agent failed"):
            self.exercise(build_error=RuntimeError("agent failed"))

    def test_runtime_error(self):
        with self.assertRaisesRegex(RuntimeError, "original computation error"):
            self.exercise(error=RuntimeError("original computation error"))


class NewtonCloseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from engines.newton_engine import NewtonEngine
            from engines.newton_recorder import NewtonVideoRecorder
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        cls.engine_type = NewtonEngine
        cls.recorder_type = NewtonVideoRecorder

    def test_closes_both_viewers_once(self):
        engine = self.engine_type.__new__(self.engine_type)
        engine._closed = False
        engine._viewer = Mock()
        engine._video_recorder = Mock()
        viewer, recorder = engine._viewer, engine._video_recorder
        engine.close()
        engine.close()
        recorder.close.assert_called_once()
        viewer.close.assert_called_once()
        self.assertIsNone(engine._viewer)

    def test_already_closed_window_is_not_closed_again(self):
        engine = self.engine_type.__new__(self.engine_type)
        engine._viewer = Mock()
        engine._viewer.renderer.window.context = None
        viewer = engine._viewer
        engine.close()
        viewer.close.assert_not_called()

    def test_partial_engine_initialization(self):
        self.engine_type.__new__(self.engine_type).close()

    def test_recorder_closes_hidden_viewer_and_releases_frames(self):
        recorder = self.recorder_type.__new__(self.recorder_type)
        recorder._viewer = Mock()
        recorder._video = Mock()
        viewer, frames = recorder._viewer, recorder._video
        recorder.close()
        recorder.close()
        viewer.close.assert_called_once()
        self.assertIsNone(recorder._viewer)
        self.assertGreaterEqual(frames.clear.call_count, 1)


class OptionalBackendCloseTests(unittest.TestCase):
    def test_gym_destroys_viewer_before_simulator_once(self):
        cls = backend_methods("isaac_gym_engine.py", "IsaacGymEngine", ["close"])
        engine = cls(False)
        engine._closed = False
        engine._gym = Mock()
        engine._viewer, engine._sim, engine._video_recorder = object(), object(), Mock()
        viewer, sim, recorder = engine._viewer, engine._sim, engine._video_recorder
        order = []
        recorder.close.side_effect = lambda: order.append("recorder")
        engine._gym.destroy_viewer.side_effect = lambda _: order.append("viewer")
        engine._gym.destroy_sim.side_effect = lambda _: order.append("sim")
        engine.close()
        engine.close()
        self.assertEqual(order, ["recorder", "viewer", "sim"])
        engine._gym.destroy_viewer.assert_called_once_with(viewer)
        engine._gym.destroy_sim.assert_called_once_with(sim)

    def test_lab_detaches_keyboard_and_closes_owned_app_once(self):
        cls = backend_methods("isaac_lab_engine.py", "IsaacLabEngine", ["close"])
        engine = cls(False)
        engine._closed = False
        engine._input, engine._keyboard, engine._sub_keyboard = Mock(), object(), object()
        engine._app_launcher, engine._video_recorder = Mock(), Mock()
        launcher, recorder, subscription = engine._app_launcher, engine._video_recorder, engine._sub_keyboard
        engine.close()
        engine.close()
        engine._input.unsubscribe_to_keyboard_events.assert_called_once_with(engine._keyboard, subscription)
        launcher.app.close.assert_called_once()
        recorder.close.assert_called_once()

    def test_optional_backends_can_close_partial_instances(self):
        for filename, name in [("isaac_gym_engine.py", "IsaacGymEngine"), ("isaac_lab_engine.py", "IsaacLabEngine")]:
            with self.subTest(name=name):
                cls = backend_methods(filename, name, ["close"])
                cls.__new__(cls).close()

    def test_lab_stops_after_window_event_closes_application(self):
        cls = backend_methods("isaac_lab_engine.py", "IsaacLabEngine", ["render"])
        engine = cls(True)
        engine._app_launcher, engine._sim, engine._draw_interface = Mock(), Mock(), Mock()
        engine._app_launcher.app.is_running.side_effect = [True, False]
        with self.assertRaises(SystemExit) as ctx:
            engine.render()
        self.assertEqual(ctx.exception.code, 0)
        engine._draw_interface.clear_lines.assert_not_called()

    def test_lab_recorder_detaches_and_destroys_render_product(self):
        cls = backend_methods("isaac_lab_recorder.py", "IsaacLabVideoRecorder", ["close"], base=MockBaseRecorder)
        recorder = cls()
        recorder._annotator, recorder._render_product = Mock(), Mock()
        annotator, product = recorder._annotator, recorder._render_product
        recorder.close()
        recorder.close()
        annotator.detach.assert_called_once_with([product])
        product.destroy.assert_called_once()


class MockBaseRecorder:
    def close(self):
        pass


class LoggerCloseTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.filename = str(Path(self.folder.name) / "log.txt")

    def test_text_logger_closes_file_and_is_reusable(self):
        log = Logger()
        self.addCleanup(log.close)
        log.configure_output_file(self.filename)
        handle = log.output_file
        log.close()
        log.close()
        self.assertTrue(handle.closed)
        log.configure_output_file(self.filename)
        self.assertFalse(log.output_file.closed)

    def test_tensorboard_writer_and_text_file_close(self):
        log = TBLogger()
        self.addCleanup(log.close)
        log.configure_output_file(self.filename)
        writer, handle = log._writer, log.output_file
        writer.add_scalar("probe", 1, 0)
        with patch.object(writer, "close", wraps=writer.close) as close:
            log.close()
            log.close()
            close.assert_called_once()
        self.assertTrue(handle.closed)
        self.assertTrue(list(Path(self.folder.name).glob("events*")))

    def test_wandb_closes_owned_run_only(self):
        owned, unrelated = Mock(), Mock()
        log = WandbLogger("probe", {})
        self.addCleanup(log.close)
        with patch("util.wandb_logger.wandb.init", return_value=owned), \
             patch("util.wandb_logger.wandb.run", unrelated):
            log.configure_output_file(self.filename)
            handle = log.output_file
            log.close()
            log.close()
        owned.finish.assert_called_once()
        unrelated.finish.assert_not_called()
        self.assertTrue(handle.closed)

    def test_writer_failure_still_closes_text_file(self):
        log = TBLogger()
        self.addCleanup(log.close)
        log.configure_output_file(self.filename)
        log._writer.close()
        log._writer = Mock()
        log._writer.close.side_effect = RuntimeError("writer failed")
        handle = log.output_file
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            log.close()
        self.assertTrue(handle.closed)

    def test_failed_logger_configuration_closes_open_file(self):
        from learning.base_agent import BaseAgent
        log = TBLogger()
        self.addCleanup(log.close)
        with patch("learning.base_agent.tb_logger.TBLogger", return_value=log), \
             patch("util.tb_logger.tensorboardX.SummaryWriter", side_effect=RuntimeError("writer init failed")):
            with self.assertRaisesRegex(RuntimeError, "writer init failed"):
                BaseAgent._build_logger(None, "tb", self.filename, {})
        self.assertIsNone(log.output_file)


class DisplayCloseTests(unittest.TestCase):
    def tearDown(self):
        display._display_process = None
        display._display_name = None

    def test_external_display_is_never_owned_or_stopped(self):
        with patch.dict(os.environ, {"DISPLAY": ":7"}), patch("util.display.subprocess.Popen") as popen:
            display.ensure_virtual_display()
            display.close_virtual_display()
            popen.assert_not_called()
            self.assertEqual(os.environ["DISPLAY"], ":7")

    def test_owned_child_is_stopped_and_environment_restored(self):
        child = Mock()
        child.poll.return_value = None
        with patch.dict(os.environ, {}, clear=True), patch("util.display.subprocess.Popen", return_value=child), \
             patch("util.display.time.sleep"):
            display.ensure_virtual_display()
            self.assertEqual(os.environ["DISPLAY"], ":99")
            display.close_virtual_display()
            display.close_virtual_display()
            self.assertNotIn("DISPLAY", os.environ)
        child.terminate.assert_called_once()
        child.wait.assert_called_once_with(timeout=5)

    def test_startup_interrupt_retains_child_for_cleanup(self):
        child = Mock()
        child.poll.return_value = None
        with patch.dict(os.environ, {}, clear=True), patch("util.display.subprocess.Popen", return_value=child), \
             patch("util.display.time.sleep", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                display.ensure_virtual_display()
            display.close_virtual_display()
        child.terminate.assert_called_once()


class EncodingAndProcessGroupTests(unittest.TestCase):
    def test_failed_encoding_still_closes_clip(self):
        clip = Mock()
        clip.write_videofile.side_effect = RuntimeError("encoder failed")
        with patch("util.video.ImageSequenceClip", return_value=clip):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                video.Video(30).save("probe.mp4")
        clip.close.assert_called_once()

    def test_process_group_shutdown_is_idempotent(self):
        with patch.object(mp_util.torch.distributed, "is_initialized", side_effect=[True, False]), \
             patch.object(mp_util.torch.distributed, "destroy_process_group") as destroy:
            mp_util.shutdown()
            mp_util.shutdown()
        destroy.assert_called_once()
        self.assertIsNone(mp_util.global_mp_device)
        self.assertEqual(mp_util.global_num_procs, 0)


class GraphicsFailureTests(unittest.TestCase):
    def test_failed_sdf_plot_save_closes_only_its_figure(self):
        import importlib.util
        import matplotlib.pyplot as plt
        spec = importlib.util.spec_from_file_location("sdf_baker_lifecycle", ROOT / "tests/Scene_Test/sdf_baker.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: module}):
            spec.loader.exec_module(module)
            unrelated = plt.figure()
            try:
                with tempfile.TemporaryDirectory() as folder:
                    with self.assertRaises(FileNotFoundError):
                        module.save_slices(np.zeros((3, 3, 3)), np.zeros(3), 0.1, "Y", 0.1,
                                           Path(folder) / "missing" / "slices.png")
                self.assertEqual(plt.get_fignums(), [unrelated.number])
            finally:
                plt.close(unrelated)

    def test_gym_releases_camera_access_even_when_read_fails(self):
        cls = backend_methods("isaac_gym_recorder.py", "IsaacGymVideoRecorder",
            ["_record_frame"], base=MockBaseRecorder, np=np, gymapi=Mock())
        recorder = cls()
        recorder._camera_ptr = 1
        recorder._env_id, recorder._obj_id = 0, 0
        recorder._cam_target, recorder._cam_pos = np.zeros(3), np.ones(3)
        recorder._set_camera_pose = Mock()
        recorder._engine = Mock()
        recorder._engine.get_root_pos.return_value = __import__('torch').zeros((1, 3))
        gym = recorder._engine.get_gym.return_value
        gym.get_camera_image.side_effect = RuntimeError("camera read failed")
        sim = recorder._engine.get_sim.return_value
        with self.assertRaisesRegex(RuntimeError, "camera read failed"):
            recorder._record_frame()
        gym.start_access_image_tensors.assert_called_once_with(sim)
        gym.end_access_image_tensors.assert_called_once_with(sim)

    def test_animation_initialization_failure_closes_only_owned_figures(self):
        import importlib.util
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        spec = importlib.util.spec_from_file_location("char_vis_util", ROOT / "tools/util/char_vis_util.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        unrelated = plt.figure()
        self.addCleanup(lambda: plt.close(unrelated))
        for function, args in [(module.vis_body_pos_anim, (np.zeros((2, 2, 3)), [-1, 0])),
                               (module.output_body_pos_anim, (np.zeros((2, 2, 3)), [-1, 0], "unused"))]:
            with patch.object(module, "plot_ani", side_effect=RuntimeError("animation failed")):
                with self.assertRaisesRegex(RuntimeError, "animation failed"):
                    function(*args)
            self.assertEqual(plt.get_fignums(), [unrelated.number])


if __name__ == "__main__":
    unittest.main()
