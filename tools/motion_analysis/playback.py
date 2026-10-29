"""Interactive PCA comparison using MimicKit's existing Newton motion viewer.

Simulator imports are deliberately deferred until playback is requested. Offline
PCA and the CLI's help/cache paths must not initialize Newton or a GL context.
"""

from pathlib import Path
import sys
from typing import Any

import numpy as np

from tools.motion_analysis.playback_controls import PlaybackControls


class PauseBridge:
    """Synchronize the environment and Newton UI without duplicate pause states."""

    def __init__(self, engine: Any):
        self.engine = engine

    def publish(self, paused: bool) -> None:
        self.engine.set_viewer_paused(paused)

    def read(self) -> bool:
        return bool(self.engine.get_viewer_paused())


class ComparisonCamera:
    """Track both rendered worlds while preserving the user's camera orbit."""

    def __init__(self, engine: Any, root: np.ndarray, world_offsets: np.ndarray,
                 obj_id: int):
        self.engine = engine
        self.obj_id = obj_id
        self.offset = np.mean(world_offsets - world_offsets[0], axis=0)
        separation = world_offsets[1] - world_offsets[0]
        back = np.cross(separation, np.array([0.0, 0.0, 1.0]))
        back /= max(float(np.linalg.norm(back)), 1e-8)
        distance = max(6.0, float(np.linalg.norm(separation)) * 1.6)
        self.target = np.asarray(root, dtype=float).copy() + self.offset
        self.target[2] = max(0.8, self.target[2])
        self.pos = self.target + back * distance + np.array([0.0, 0.0, distance * 0.5])
        self.engine.set_camera_pose(self.pos, self.target)

    def update(self) -> None:
        camera_pos = self.engine.get_camera_pos()
        camera_dir = self.engine.get_camera_dir()
        previous_delta = self.target - self.pos
        distance = float(np.linalg.norm(previous_delta))
        previous_dir = previous_delta / max(distance, 1e-8)
        if not np.allclose(camera_pos, self.pos, atol=1e-5):
            delta = camera_pos - self.target
        elif not np.allclose(camera_dir, previous_dir, atol=1e-5):
            delta = -distance * camera_dir
        else:
            delta = camera_pos - self.target

        roots = self.engine.get_root_pos(self.obj_id).detach().cpu().numpy()
        target = np.mean(roots, axis=0) + self.offset
        target[2] = self.target[2]
        self.target = target
        self.pos = target + delta
        self.engine.set_camera_pose(self.pos, self.target)


def _environment_types():
    """Build the small adapters only after the caller explicitly requests GL."""
    import torch

    import anim.motion_lib as motion_lib
    import envs.sim_env as sim_env
    import envs.view_motion_env as view_motion_env

    class InMemoryMotionLib(motion_lib.MotionLib):
        """Reuse all standard interpolation/FK preparation with two memory clips."""

        def __init__(self, source: Any, reconstructed: np.ndarray, model: Any, device: str):
            self._source = source
            self._reconstructed = reconstructed
            super().__init__(str(source.path), model, device)

        def _load_motion_pkl(self, motion_file):
            self._motion_files = [str(self._source.path), "<PCA reconstruction>"]
            self._motion_weights = torch.ones(2, dtype=torch.float32, device=self._device)
            self._motion_fps = torch.full(
                (2,), float(self._source.fps), dtype=torch.float32, device=self._device)
            self._motion_num_frames = torch.tensor(
                [len(self._source.frames)] * 2, dtype=torch.long, device=self._device)
            self._motion_loop_modes = torch.full(
                (2,), int(self._source.loop_mode), dtype=torch.int, device=self._device)
            poses = [self._extract_frame_data(frames)
                     for frames in (self._source.frames, self._reconstructed)]
            self._frame_root_pos = torch.cat([pose[0] for pose in poses], dim=0)
            self._frame_root_rot = torch.cat([pose[1] for pose in poses], dim=0)
            self._frame_joint_rot = torch.cat([pose[2] for pose in poses], dim=0)

    class PcaPlaybackEnv(view_motion_env.ViewMotionEnv):
        """Add PCA selection controls while retaining the ordinary playback logic."""

        def __init__(self, session, controls, env_config, engine_config, device):
            self.session = session
            self.controls = controls
            self._reconstructed = self._checked_reconstruction(controls.active_indices)
            self._spacing = float(engine_config["env_spacing"])
            super().__init__(env_config, engine_config, 2, device, True)
            self._pause_bridge = PauseBridge(self._engine)
            self._pause_bridge.publish(False)

        def _checked_reconstruction(self, indices):
            frames = np.asarray(self.session.reconstruct(indices), dtype=np.float32)
            source = np.asarray(self.session.source.frames)
            if frames.shape != source.shape or not np.isfinite(frames).all():
                raise ValueError("PCA reconstruction must contain finite frames matching the source shape.")
            if not np.allclose(frames[:, :6], source[:, :6], rtol=0, atol=1e-6):
                raise ValueError("PCA reconstruction must preserve the source root pose.")
            return frames

        def _load_motions(self, motion_file):
            self._motion_lib = InMemoryMotionLib(
                self.session.source, self._reconstructed, self._kin_char_model, self._device)

        def _build_character(self, env_id, env_config, color=None):
            colors = (np.array([0.15, 0.45, 0.9]), np.array([0.95, 0.45, 0.1]))
            return super()._build_character(env_id, env_config, color=colors[env_id])

        def _build_camera(self, env_config):
            from newton.utils import compute_world_offsets

            offsets = np.asarray(compute_world_offsets(
                2, (self._spacing, self._spacing, 0.0), self._engine.get_sim_model().up_axis))
            self._camera = ComparisonCamera(
                self._engine, self.session.source.frames[0, :3], offsets, self._get_char_id())

        def _setup_gui(self):
            def shortcut(action):
                def callback():
                    if not self.controls.captures_keyboard:
                        self.controls.request_action(action)
                return callback

            self._engine.register_keyboard_callback("ENTER", shortcut("toggle"))
            self._engine.register_keyboard_callback("SPACE", shortcut("step"))
            self._engine.register_ui_callback(self.controls.draw, position="side")

        def reset(self, env_ids=None):
            # A comparison always resets both characters on the same timeline.
            super().reset(None)
            self._sync_motion()
            self._update_observations()
            self._update_info()
            return self._obs_buf, self._info

        def _set_play_mode(self, mode):
            self._play_mode = mode
            self.controls.paused = mode != sim_env.PlayMode.PLAY
            self._pause_bridge.publish(self.controls.paused)

        def _apply_actions(self, actions):
            for action in actions:
                if action == "restart":
                    self.reset()
                elif action == "step":
                    self._set_play_mode(sim_env.PlayMode.ONE_STEP)
                elif action == "toggle":
                    mode = (sim_env.PlayMode.PLAY if self._play_mode == sim_env.PlayMode.PAUSE
                            else sim_env.PlayMode.PAUSE)
                    self._set_play_mode(mode)
                else:
                    mode = sim_env.PlayMode.PLAY if action == "play" else sim_env.PlayMode.PAUSE
                    self._set_play_mode(mode)

        def _apply_selection(self):
            indices = self.controls.take_selection()
            if indices is None or indices == self.controls.active_indices:
                return
            try:
                reconstructed = self._checked_reconstruction(indices)
                library = InMemoryMotionLib(
                    self.session.source, reconstructed, self._kin_char_model, self._device)
            except (ValueError, RuntimeError, OSError) as exc:
                self.controls.error = str(exc)
                return
            self._reconstructed = reconstructed
            self._motion_lib = library
            self.controls.selection_applied(indices)
            self._sync_motion()

        def _save_selection(self):
            if self.controls.take_save_request():
                try:
                    path = self.session.save_reconstruction(self.controls.active_indices)
                    self.controls.status = f"Saved: {path}"
                    self.controls.error = ""
                except (ValueError, RuntimeError, OSError) as exc:
                    self.controls.error = str(exc)

        def _render_scene(self):
            self.controls.paused = self._play_mode != sim_env.PlayMode.PLAY
            self.controls.time = float(self._time_buf[0].item())
            self._pause_bridge.publish(self.controls.paused)
            super()._render_scene()

            actions = self.controls.take_actions()
            if actions:
                # Viewer handles SPACE first; publish ONE_STEP's paused flag after it.
                self._apply_actions(actions)
            elif self._play_mode != sim_env.PlayMode.ONE_STEP:
                mode = (sim_env.PlayMode.PAUSE if self._pause_bridge.read()
                        else sim_env.PlayMode.PLAY)
                self._play_mode = mode
                self.controls.paused = mode != sim_env.PlayMode.PLAY

            # These run for every render, including the parent's paused render loop.
            self._apply_selection()
            self._save_selection()

    return InMemoryMotionLib, PcaPlaybackEnv


def play_session(session: Any, initial_indices: tuple[int, ...], *,
                 selection_mode: str = "cumulative", env_config: Path,
                 engine_config: Path, device: str = "cuda:0") -> None:
    """Play source and selected PCA reconstruction with one shared clock."""
    root = Path(__file__).resolve().parents[2]
    if str(root / "mimickit") not in sys.path:
        sys.path.insert(0, str(root / "mimickit"))

    import torch
    from envs.env_builder import load_configs

    environment, engine = load_configs(str(env_config), str(engine_config))
    if engine.get("engine_name") != "newton":
        raise ValueError("Interactive PCA playback currently requires the Newton engine.")
    environment = dict(environment)
    environment["motion_file"] = str(session.source.path)
    controls = PlaybackControls(session, tuple(initial_indices), selection_mode)
    _, environment_type = _environment_types()
    env = None
    try:
        # Retain the instance if construction fails after opening its GL window.
        env = environment_type.__new__(environment_type)
        env.__init__(session, controls, environment, dict(engine), device)
        env.reset()
        action = torch.zeros((2, *env.get_action_space().shape), device=device)
        with torch.no_grad():
            while True:
                _, _, done, _ = env.step(action)
                if bool(torch.any(done != 0)):
                    env.reset()
    finally:
        if env is not None:
            env.close()
