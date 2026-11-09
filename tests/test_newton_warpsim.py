"""Real newton_1_7 / WarpSim checks, without training or rewriting robot assets.

Run with MIMICKIT_TEST_NEWTON=1. Set MIMICKIT_G1_ASSET to an existing G1 XML
and MIMICKIT_TEST_NEWTON_GL=1 to include OpenGL capture. All generated files
and the Warp cache stay under tmp/warpsim-validation in this checkout.
"""
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
SCRATCH = ROOT / "tmp" / "warpsim-validation"


@unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON") == "1", "Opt in to GPU Newton checks.")
class WarpSimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global np, torch, wp, newton, build_engine, ObjType
        SCRATCH.mkdir(parents=True, exist_ok=True)
        os.environ["WARP_CACHE_PATH"] = str(SCRATCH / "warp-cache")
        import numpy as np
        import torch
        import warp as wp
        import newton
        from engines.engine_builder import build_engine
        from engines.engine import ObjType
        wp.init()
        if not wp.is_cuda_available():
            raise unittest.SkipTest("CUDA device required")
        wp.set_device("cuda:0")
        cls.temp = tempfile.TemporaryDirectory(dir=SCRATCH)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.hinge = Path(cls.temp.name) / "hinge.xml"
        cls.ball = Path(cls.temp.name) / "ball.xml"
        cls.sphere = Path(cls.temp.name) / "sphere.xml"
        for path, joints in [(cls.hinge, '<joint name="hinge" type="hinge" axis="0 1 0"/>'),
                             (cls.ball, '<joint name="ball" type="ball"/>')]:
            path.write_text('''<mujoco><compiler angle="radian"/>
              <default><joint stiffness="20" damping="2" armature="0.1"
                actuatorfrcrange="-3 3"/></default>
              <worldbody><body name="base"><freejoint/>
                <geom type="sphere" size="0.1" mass="1"/>
                <body name="tip" pos="0 0 0.3">''' + joints + '''
                  <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.04" mass="0.2"/>
                </body></body></worldbody></mujoco>''')
        cls.sphere.write_text('''<mujoco><worldbody><body name="ball"><freejoint/>
            <geom type="sphere" size="0.1" mass="1"/>
            </body></worldbody></mujoco>''')

    def build(self, assets, mode="pos", worlds=2, graph=True, positions=None,
              fixed=False, sim_freq=60, record=False, visualize=False, color=None):
        config = dict(engine_name="newton_1_7", control_mode=mode, control_freq=30, sim_freq=sim_freq,
                      env_spacing=5, newton_njmax=1024, newton_nconmax=256)
        e = build_engine(config, worlds, "cuda:0", visualize, record_video=record)
        self.addCleanup(e.close)
        self.assertEqual(e.get_name(), "newton_1_7")
        for world in range(worlds):
            e.create_env()
            for obj, asset in enumerate(assets):
                pos = positions[obj] if positions is not None else [obj * 3, 0, 3]
                e.create_obj(world, ObjType.articulated, str(asset), str(obj),
                             fix_root=fixed, start_pos=pos, color=color)
        if graph:
            e.initialize_sim()
        else:
            with patch.object(e, "_build_graphs", lambda: setattr(e, "_graph", None)):
                e.initialize_sim()
        return e

    def assert_finite(self, e):
        for obj in range(e.get_objs_per_env()):
            for value in (e.get_dof_pos(obj), e.get_dof_vel(obj), e.get_dof_forces(obj),
                          e.get_root_pos(obj), e.get_body_pos(obj),
                          e.get_contact_forces(obj), e.get_ground_contact_forces(obj)):
                self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertEqual(e.get_dof_forces(obj).shape, e.get_dof_vel(obj).shape)

    def test_target_layout_world_and_object_isolation(self):
        e = self.build([self.hinge, self.ball], sim_freq=30)
        scalar = torch.tensor([[0.2], [-0.35]], device="cuda:0")
        spherical = torch.tensor([[0.1, -0.2, 0.3], [-0.3, 0.1, 0.2]], device="cuda:0")
        e.set_cmd(0, scalar)
        e.set_cmd(1, spherical)
        e._update_control_targets()
        model = e.get_sim_model()
        starts, qstart = model.articulation_start.numpy(), model.joint_q_start.numpy()
        targets = e._controls.control.joint_target_q.numpy()
        for world in range(2):
            self.assertAlmostEqual(targets[qstart[starts[world * 2] + 1]], scalar[world, 0].item())
            p = spherical[world].cpu().numpy()
            angle = np.linalg.norm(p)
            expected = np.r_[p / angle * np.sin(angle / 2), np.cos(angle / 2)]
            q = qstart[starts[world * 2 + 1] + 1]
            np.testing.assert_allclose(targets[q:q + 4], expected, atol=1e-6)
        e.set_dof_pos(None, 1, spherical)
        e._sim_state.pre_step_update()
        e._sim_state.post_step_update()
        torch.testing.assert_close(e.get_dof_pos(1), spherical)
        e.set_cmd(1, torch.zeros_like(spherical))
        e._update_control_targets()
        self.assertTrue(np.isfinite(e._controls.control.joint_target_q.numpy()).all())
        # Check the first substep: subsequent PD braking can reverse torque.
        e.step()
        self.assertGreater(e.get_dof_forces(0)[0, 0].item(), 0)
        self.assertLess(e.get_dof_forces(0)[1, 0].item(), 0)
        for _ in range(3):
            e.step()
        self.assert_finite(e)
        e.set_dof_pos(torch.tensor([1], device="cuda:0"), 0, scalar[1:])
        e.step()
        self.assert_finite(e)

    def test_all_control_modes_fixed_and_floating(self):
        for mode in ("none", "pos", "vel", "torque", "pd_explicit"):
            for fixed in (False, True):
                with self.subTest(mode=mode, fixed=fixed):
                    e = self.build([self.hinge, self.ball], mode=mode, fixed=fixed, worlds=1)
                    for obj in range(2):
                        e.set_cmd(obj, torch.full_like(e.get_dof_vel(obj), 0.15))
                    for _ in range(4):
                        e.step()
                    self.assert_finite(e)
                    if mode == "none":
                        self.assertEqual(e.get_dof_forces(0).abs().max().item(), 0)
                    if mode == "torque":
                        torch.testing.assert_close(e.get_dof_forces(0), torch.full_like(e.get_dof_vel(0), 0.15))
                    if mode in ("pos", "vel"):
                        self.assertGreater(e.get_dof_forces(0).abs().max().item(), 0)
                    e.close()

    def test_rotated_ball_velocity_frames_and_partial_reset(self):
        # A local X rotation at a 90-degree Z pose must become parent-frame Y.
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                e = self.build([self.hinge, self.ball], mode="none", fixed=fixed)
                pose = torch.tensor([[0., 0., np.pi / 2], [0., 0., -np.pi / 2]], device="cuda:0")
                local = torch.tensor([[1., 0., 0.], [2., 0., 0.]], device="cuda:0")
                parent = torch.tensor([[0., 1., 0.], [0., -2., 0.]], device="cuda:0")
                hinge = torch.tensor([[0.3], [-0.4]], device="cuda:0")
                e.set_dof_pos(None, 1, pose)
                e.set_dof_vel(None, 1, local)
                e.set_dof_vel(None, 0, hinge)
                if not fixed:
                    e.set_root_vel(None, 1, torch.tensor([0.2, 0., 0.], device="cuda:0"))
                state = e._sim_state
                state.pre_step_update()
                raw = state._torch_joint_qd.view(2, -1)
                model = e.get_sim_model()
                j = model.articulation_start.numpy()[1] + 1
                d = model.joint_qd_start.numpy()[j]
                torch.testing.assert_close(raw[:, d:d + 3], parent, atol=1e-6, rtol=0)
                state.eval_fk()
                torch.testing.assert_close(e.get_body_ang_vel(1)[:, -1], parent, atol=1e-6, rtol=0)
                state.post_step_update()
                torch.testing.assert_close(e.get_dof_vel(1), local, atol=1e-6, rtol=0)
                torch.testing.assert_close(e.get_dof_vel(0), hinge)
                if not fixed:
                    torch.testing.assert_close(e.get_root_vel(1)[:, 0], torch.full((2,), 0.2, device="cuda:0"))
                # Reset one world, including the scalar/broadcast setter path.
                e.set_dof_pos(torch.tensor([1], device="cuda:0"), 1, torch.zeros((1, 3), device="cuda:0"))
                e.set_dof_vel(torch.tensor([1], device="cuda:0"), 1, 0.0)
                state.pre_step_update()
                state.post_step_update()
                torch.testing.assert_close(e.get_dof_vel(1)[0], local[0], atol=1e-6, rtol=0)
                torch.testing.assert_close(e.get_dof_vel(1)[1], torch.zeros(3, device="cuda:0"))
                e.close()

    def test_rotated_ball_velocity_torque_and_pd_commands(self):
        for mode in ("vel", "torque", "pd_explicit"):
            with self.subTest(mode=mode):
                e = self.build([self.ball], mode=mode, worlds=1, fixed=True)
                pose = torch.tensor([[0., 0., np.pi / 2]], device="cuda:0")
                local = torch.tensor([[0.3, 0., 0.]], device="cuda:0")
                e.set_dof_pos(None, 0, pose)
                e.set_dof_vel(None, 0, local if mode == "pd_explicit" else 0.0)
                e.set_cmd(0, pose if mode == "pd_explicit" else local)
                state, control = e._sim_state, e._controls.control
                state.pre_step_update()
                e._pre_sim_step(state.raw_state, control)
                model = e.get_sim_model()
                d = model.joint_qd_start.numpy()[1]
                if mode == "pd_explicit":
                    kd = float(e._kd_raw.numpy()[d])
                    expected = np.array([0., -kd * 0.3, 0.])
                    torch.testing.assert_close(e.get_dof_forces(0), -kd * local, atol=1e-5, rtol=0)
                else:
                    expected = np.array([0., 0.3, 0.])
                raw = control.joint_target_qd if mode == "vel" else control.joint_f
                np.testing.assert_allclose(raw.numpy()[d:d + 3], expected, atol=1e-5)
                # Repeated substeps must rotate the original command, not accumulate rotations.
                e._pre_sim_step(state.raw_state, control)
                np.testing.assert_allclose(raw.numpy()[d:d + 3], expected, atol=1e-5)
                e.step()
                self.assert_finite(e)
                if mode == "vel":
                    # MuJoCo's BALL generalized torque is in the rotating body
                    # frame (the fixture's child anchor has identity rotation).
                    np.testing.assert_allclose(e.get_dof_forces(0).cpu().numpy(),
                                               e._solver.mjw_data.qfrc_actuator.numpy(),
                                               atol=1e-5, rtol=1e-5)
                e.close()

    def test_pd_zero_error_response_and_limits_exclude_roots(self):
        e = self.build([self.hinge, self.ball], mode="pd_explicit")
        for obj in range(2):
            q = torch.full_like(e.get_dof_vel(obj), 0.1)
            e.set_dof_pos(None, obj, q)
            e.set_dof_vel(None, obj, torch.zeros_like(q))
            e.set_cmd(obj, q)
        e._sim_state.pre_step_update()
        e._apply_pd_explicit_torque(e._sim_state.raw_state, e._controls.control)
        for obj in range(2):
            torch.testing.assert_close(e.get_dof_forces(obj), torch.zeros_like(e.get_dof_vel(obj)), atol=1e-5, rtol=0)
            e.set_cmd(obj, torch.full_like(e.get_dof_vel(obj), 0.15))
        e._apply_pd_explicit_torque(e._sim_state.raw_state, e._controls.control)
        for obj in range(2):
            torch.testing.assert_close(e.get_dof_forces(obj), torch.ones_like(e.get_dof_vel(obj)), atol=1e-5, rtol=0)
            e.set_cmd(obj, torch.full_like(e.get_dof_vel(obj), 1e8))
        e._apply_pd_explicit_torque(e._sim_state.raw_state, e._controls.control)
        for obj in range(2):
            # Native BALL import does not use the scalar joint actuatorfrcrange.
            # Clamp against the imported model, without retuning its limits.
            limit = torch.as_tensor(e.get_obj_torque_limits(0, obj), device="cuda:0")
            torch.testing.assert_close(e.get_dof_forces(obj), limit.expand_as(e.get_dof_vel(obj)))
        np.testing.assert_array_equal(e.get_obj_torque_limits(0, 0), [3])
        force = e._controls.control.joint_f.numpy()
        qd = e._sim_model.joint_qd_start.numpy()
        for root in e._sim_model.articulation_start.numpy()[:-1]:
            np.testing.assert_array_equal(force[qd[root]:qd[root + 1]], 0)

    def test_ground_other_contacts_and_no_contact(self):
        e = self.build([self.sphere] * 4, mode="none", sim_freq=30,
                       positions=[[0, 0, 0.095], [2, 0, 2], [2.18, 0, 2], [5, 0, 2]])
        e.step()
        floor = e.get_ground_contact_forces(0)
        self.assertTrue(bool((floor[:, 0, 2] > 0).all()))
        torch.testing.assert_close(e.get_contact_forces(0), floor)
        for obj in (1, 2):
            self.assertGreater(e.get_contact_forces(obj).abs().max().item(), 0)
            self.assertEqual(e.get_ground_contact_forces(obj).abs().max().item(), 0)
        torch.testing.assert_close(e.get_contact_forces(1), -e.get_contact_forces(2))
        self.assertEqual(e.get_contact_forces(3).abs().max().item(), 0)
        self.assert_finite(e)

    def test_graph_matches_eager_and_reset_odd_even_substeps(self):
        for sim_freq in (30, 60):
            with self.subTest(sim_freq=sim_freq):
                positions = [[0, 0, 0.095], [3, 0, 3]]
                graph = self.build([self.hinge, self.ball], sim_freq=sim_freq, positions=positions)
                eager = self.build([self.hinge, self.ball], graph=False, sim_freq=sim_freq, positions=positions)
                for frame in range(10):
                    for e in (graph, eager):
                        if frame == 5:
                            for obj in range(2):
                                e.set_root_pos(None, obj, torch.tensor([obj * 3, 0, 3], device="cuda:0"))
                                e.set_root_rot(None, obj, torch.tensor([0, 0, 0, 1], device="cuda:0"))
                                e.set_root_vel(None, obj, torch.zeros((2, 3), device="cuda:0"))
                                e.set_root_ang_vel(None, obj, torch.zeros((2, 3), device="cuda:0"))
                                e.set_dof_pos(None, obj, torch.full_like(e.get_dof_pos(obj), 0.2))
                                e.set_dof_vel(None, obj, torch.full_like(e.get_dof_vel(obj), 0.1))
                        for obj in range(2):
                            e.set_cmd(obj, torch.full_like(e.get_dof_pos(obj), 0.02 * frame))
                        e.step()
                        self.assert_finite(e)
                    for obj in range(2):
                        for getter in ("get_body_pos", "get_dof_pos", "get_dof_vel", "get_dof_forces",
                                       "get_contact_forces", "get_ground_contact_forces"):
                            torch.testing.assert_close(getattr(graph, getter)(obj), getattr(eager, getter)(obj),
                                                       rtol=1e-4, atol=2e-5)
                graph.close()
                eager.close()

    def test_external_force_is_one_step_and_world_local(self):
        e = self.build([self.sphere], mode="none", sim_freq=60)
        e.set_body_forces(torch.tensor([1], device="cuda:0"), 0, 0,
                          torch.tensor([[3.0, 0, 0]], device="cuda:0"))
        e.step()
        vx = e.get_root_vel(0)[:, 0].clone()
        torch.testing.assert_close(vx, torch.tensor([0.0, 0.1], device="cuda:0"), atol=1e-5, rtol=0)
        e.step()
        torch.testing.assert_close(e.get_root_vel(0)[:, 0], vx, atol=1e-5, rtol=0)

    def test_humanoid_asset(self):
        e = self.build([ROOT / "data/assets/humanoid/humanoid.xml"])
        self.assertEqual(e.get_dof_pos(0).shape, (2, 28))
        self.assertTrue(e._sim_state._has_spherical_joints())
        for _ in range(8):
            e.set_cmd(0, torch.zeros_like(e.get_dof_pos(0)))
            e.step()
        self.assert_finite(e)

    def test_g1_asset_and_authored_physics(self):
        asset = Path(os.environ.get("MIMICKIT_G1_ASSET", ROOT / "data/assets/g1/g1.xml"))
        if not asset.is_file():
            self.skipTest("G1 XML absent; set MIMICKIT_G1_ASSET to an existing asset")
        e = self.build([asset], sim_freq=240)
        self.assertEqual(e.get_dof_pos(0).shape, (2, 29))
        self.assertEqual(e.get_timestep(), 1 / 30)
        self.assertEqual(e.get_sim_timestep(), 1 / 240)
        xml = ET.parse(asset)
        expected_mass = sum(float(x.attrib["mass"]) for x in xml.findall(".//inertial"))
        self.assertAlmostEqual(e.calc_obj_mass(0, 0), expected_mass, places=4)
        joints = {x.attrib["name"]: x.attrib for x in xml.findall(".//worldbody//joint")}
        model = e.get_sim_model()
        matched = 0
        for j, label in enumerate(model.joint_label):
            name = label.split("/")[-1]
            if name not in joints:
                continue
            matched += 1
            source = joints[name]
            d = model.joint_qd_start.numpy()[j]
            self.assertAlmostEqual(model.joint_target_ke.numpy()[d], float(source["stiffness"]), places=4)
            self.assertAlmostEqual(model.joint_target_kd.numpy()[d], float(source["damping"]), places=5)
            self.assertAlmostEqual(model.joint_armature.numpy()[d], float(source["armature"]), places=6)
            low, high = map(float, source["range"].split())
            self.assertAlmostEqual(model.joint_limit_lower.numpy()[d], low, places=5)
            self.assertAlmostEqual(model.joint_limit_upper.numpy()[d], high, places=5)
            self.assertAlmostEqual(model.joint_effort_limit.numpy()[d],
                                   max(abs(float(x)) for x in source["actuatorfrcrange"].split()), places=4)
        self.assertEqual(matched, 58)
        inertials = {b.attrib["name"]: b.find("inertial") for b in xml.findall(".//worldbody//body")}
        matched = 0
        for b, label in enumerate(model.body_label):
            source = inertials.get(label.split("/")[-1])
            if source is None:
                continue
            expected = sorted(map(float, source.attrib["diaginertia"].split()))
            np.testing.assert_allclose(np.linalg.eigvalsh(model.body_inertia.numpy()[b]), expected,
                                       rtol=1e-4, atol=1e-7)
            matched += 1
        self.assertEqual(matched, model.body_count)
        np.testing.assert_array_equal(model.joint_damping.numpy(), 0)
        np.testing.assert_array_equal(model.mujoco.dof_passive_stiffness.numpy(), 0)
        for _ in range(8):
            e.set_cmd(0, torch.zeros_like(e.get_dof_pos(0)))
            e.step()
        self.assert_finite(e)

    @unittest.skipUnless(os.environ.get("MIMICKIT_TEST_NEWTON_GL") == "1", "Opt in to OpenGL checks.")
    def test_viewer_and_video(self):
        e = self.build([ROOT / "data/assets/humanoid/humanoid.xml"], worlds=1, record=True,
                       positions=[[0, 0, 1]], color=[0.2, 0.7, 0.3])
        model = e.get_sim_model()
        expected_colors = model.shape_color.numpy()
        body_shapes = model.body_shapes[0]
        np.testing.assert_allclose(expected_colors[body_shapes],
                                   np.tile([0.2, 0.7, 0.3], (len(body_shapes), 1)), atol=1e-6)
        e.set_body_color(0, 0, 0, [0.8, 0.1, 0.2])
        expected_colors[body_shapes] = [0.8, 0.1, 0.2]
        np.testing.assert_allclose(model.shape_color.numpy(), expected_colors, atol=1e-6)
        e.start_video_recording()
        for _ in range(3):
            e.step()
        e.stop_video_recording()
        frames = e.get_video_recording().get_frames()
        self.assertEqual(len(frames), 3)
        for frame in frames:
            self.assertEqual(frame.shape, (480, 854, 3))
            self.assertGreater(np.std(frame), 1)
        e.get_video_recording().save(str(SCRATCH / "humanoid-smoke.mp4"))
        self.assertGreater((SCRATCH / "humanoid-smoke.mp4").stat().st_size, 1000)
        e.close()
        # Closing a pyglet window sets the process-wide exit flag. Exercise a
        # normal interactive launch in a fresh process, as the CLI would.
        subprocess.run([sys.executable, "-B", str(Path(__file__)), "--viewer-probe"], check=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--viewer-probe"]:
        WarpSimTests.setUpClass()
        case = WarpSimTests()
        try:
            visible = case.build([case.sphere], worlds=1, visualize=True, color=[0.2, 0.7, 0.3])
            visible.render()
            visible.close()
            visible.close()
        finally:
            case.doCleanups()
            WarpSimTests.doClassCleanups()
    else:
        unittest.main(verbosity=2)
