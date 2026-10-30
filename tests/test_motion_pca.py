"""PCA, temporal reconstruction and cache contracts without a simulator."""

import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.motion_analysis.data.motion import discontinuity_diagnostics, read_motion
from tools.motion_analysis.core.pca import fit_pca
from tools.motion_analysis.reporting.reconstruction import generate_report
from tools.motion_analysis.core.samples import build_samples, restore_samples
from tools.motion_analysis.core.selection import parse_components, resolve_selection
from tools.motion_analysis.core.session import load_session


def write_motion(path: Path, features: np.ndarray, fps: float = 60.0, loop_mode: int = 0) -> None:
    frames = np.zeros((len(features), 6 + features.shape[1]))
    frames[:, :3] = np.arange(len(features))[:, None] * .01
    frames[:, 6:] = features
    path.write_bytes(pickle.dumps({"frames": frames, "fps": fps, "loop_mode": loop_mode}))


class PCAContracts(unittest.TestCase):
    def setUp(self):
        self.samples = np.random.default_rng(47).normal(size=(31, 5)) + np.arange(5)

    def test_full_rank_reconstruction_and_variance(self):
        model = fit_pca(self.samples)
        scores = model.transform(self.samples)
        reconstructed = model.inverse_transform(scores, tuple(range(model.max_components)))
        np.testing.assert_allclose(reconstructed, self.samples, atol=2e-14)
        self.assertAlmostEqual(float(model.variance_ratio.sum()), 1.0)
        self.assertTrue(np.all(np.diff(model.variance) <= 0))
        np.testing.assert_allclose(model.components @ model.components.T, np.eye(5), atol=1e-14)

    def test_selection_adds_mean_once_and_error_is_monotonic(self):
        model = fit_pca(self.samples)
        scores = model.transform(self.samples)
        contributions = [model.inverse_transform(scores, (i,)) - model.mean for i in range(3)]
        expected = model.mean + sum(contributions)
        np.testing.assert_allclose(expected, model.inverse_transform(scores, (0, 1, 2)), atol=1e-14)
        errors = [np.mean((self.samples - model.inverse_transform(scores, tuple(range(k)))) ** 2)
                  for k in range(6)]
        self.assertTrue(np.all(np.diff(errors) <= 1e-14))

    def test_constant_samples_are_mean_only(self):
        samples = np.full((13, 5), .1)
        model = fit_pca(samples)
        self.assertEqual(model.max_components, 0)
        self.assertEqual(model.variance_ratio.shape, (0,))
        restored = model.inverse_transform(model.transform(samples), ())
        np.testing.assert_array_equal(np.broadcast_to(restored, samples.shape), samples)

    def test_invalid_pca_matrices(self):
        for samples in (np.ones(3), np.ones((1, 3)), np.ones((3, 0)),
                        np.array([[np.nan], [0.]]), np.array([[np.inf], [0.]])):
            with self.subTest(shape=samples.shape), self.assertRaises(ValueError):
                fit_pca(samples)

    def test_pose_pca_is_invariant_to_frame_permutation(self):
        original = fit_pca(self.samples)
        shuffled = fit_pca(self.samples[::-1])
        np.testing.assert_allclose(original.variance, shuffled.variance, atol=1e-14)
        np.testing.assert_allclose(original.components.T @ original.components,
                                   shuffled.components.T @ shuffled.components, atol=1e-14)


class WindowContracts(unittest.TestCase):
    def setUp(self):
        self.features = np.arange(23 * 3, dtype=float).reshape(23, 3)

    def test_tail_coverage_and_lossless_overlap_add(self):
        samples, layout = build_samples(self.features, "window", 10, 4)
        self.assertEqual(layout.starts.tolist(), [0, 4, 8, 12, 13])
        np.testing.assert_array_equal(restore_samples(samples, layout), self.features)

    def test_stride_equal_window_still_covers_tail(self):
        samples, layout = build_samples(self.features, "window", 6, 6)
        self.assertEqual(layout.starts.tolist(), [0, 6, 12, 17])
        np.testing.assert_array_equal(restore_samples(samples, layout), self.features)

    def test_window_one_equals_pose(self):
        pose, _ = build_samples(self.features)
        windows, layout = build_samples(self.features, "window", 1, 1)
        np.testing.assert_array_equal(pose, windows)
        np.testing.assert_array_equal(restore_samples(windows, layout), self.features)

    def test_invalid_window_parameters(self):
        for window, stride in ((24, 1), (23, 1), (0, 1), (10, 0), (10, 11), (2.5, 1)):
            with self.subTest(window=window, stride=stride), self.assertRaises(ValueError):
                build_samples(self.features, "window", window, stride)


class SelectionContracts(unittest.TestCase):
    def test_user_input_and_three_modes(self):
        self.assertEqual(parse_components("3, 1,3 2"), (1, 2, 3))
        self.assertEqual(parse_components("1,3,5-8,6"), (1, 3, 5, 6, 7, 8))
        self.assertEqual(parse_components(""), ())
        self.assertEqual(resolve_selection("cumulative", 3, None, 5), (0, 1, 2))
        self.assertEqual(resolve_selection("single", 3, None, 5), (2,))
        self.assertEqual(resolve_selection("subset", None, (3, 1, 3), 5), (0, 2))
        self.assertEqual(resolve_selection("cumulative", 0, None, 5), ())
        self.assertEqual(resolve_selection("subset", None, (), 0), ())

    def test_invalid_selections(self):
        for text in ("0", "-1", "1.5", "one", "3-1", "0-2", "1-10000000000"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_components(text)
        for arguments in (("single", 0, None, 5), ("single", 6, None, 5),
                          ("cumulative", 6, None, 5), ("subset", None, (0,), 5),
                          ("subset", None, (1.5,), 5), ("invalid", 2, None, 5)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                resolve_selection(*arguments)


class SessionContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.pkl"
        self.cache = self.root / "cache"
        self.features = np.random.default_rng(23).normal(size=(23, 5))
        write_motion(self.source, self.features, loop_mode=1)

    def session(self, **kwargs):
        return load_session(self.source, cache_root=self.cache, **kwargs)

    def test_cache_hit_avoids_svd_and_preserves_source_metadata(self):
        first = self.session()
        self.assertFalse(first.fit_cache_hit)
        with patch("tools.motion_analysis.core.session.fit_pca", side_effect=AssertionError("Refitted")):
            second = self.session()
        self.assertTrue(second.fit_cache_hit)
        restored = second.reconstruct(tuple(range(second.max_components)))
        np.testing.assert_allclose(restored, second.source.frames, atol=1e-14)
        np.testing.assert_array_equal(restored[:, :6], second.source.frames[:, :6])
        output = read_motion(second.save_reconstruction((0, 2)))
        self.assertEqual(output.fps, 60.)
        self.assertEqual(output.loop_mode, 1)
        self.assertEqual(output.frames.shape, second.source.frames.shape)

    def test_preview_is_memory_only_and_indices_share_cache(self):
        session = self.session()
        session.reconstruct((1, 0))
        self.assertEqual(list(session.cache_dir.glob("*.pkl")), [])
        first = session.save_reconstruction((1, 0, 0))
        second = session.save_reconstruction((0, 1))
        self.assertEqual(first, second)
        self.assertEqual(len(list(session.cache_dir.glob("*.pkl"))), 1)
        with patch.object(session, "reconstruct", side_effect=AssertionError("Reconstructed")):
            self.assertEqual(session.save_reconstruction((0, 1)), first)

    def test_saved_reconstruction_is_read_without_inverse_pca(self):
        original = self.session()
        output = original.save_reconstruction((0, 2))
        cached = self.session()
        with patch("tools.motion_analysis.core.pca.PCAModel.inverse_transform",
                   side_effect=AssertionError("Repeated inverse PCA")):
            frames = cached.reconstruct((2, 0))
            np.testing.assert_array_equal(frames, read_motion(output).frames)
            np.testing.assert_array_equal(frames[:, :6], cached.source.frames[:, :6])

    def test_cache_identity_changes_for_content_mode_window_and_version(self):
        first = self.session()
        window = self.session(mode="window", window=4)
        stride = self.session(mode="window", window=4, stride=2)
        version = self.session(version="v2")
        self.assertEqual(len({s.cache_dir for s in (first, window, stride, version)}), 4)
        write_motion(self.source, self.features + 1)
        changed = self.session()
        self.assertNotEqual(first.cache_dir, changed.cache_dir)
        self.assertFalse(changed.fit_cache_hit)

    def test_missing_or_corrupt_fit_recomputes(self):
        first = self.session()
        (first.cache_dir / "pca_model.npz").write_bytes(b"incomplete")
        self.assertFalse(self.session().fit_cache_hit)
        (first.cache_dir / "manifest.json").write_text("[]", encoding="utf-8")
        self.assertFalse(self.session().fit_cache_hit)
        (first.cache_dir / "manifest.json").unlink()
        self.assertFalse(self.session().fit_cache_hit)

    def test_corrupt_reconstruction_recomputes_and_force_is_explicit(self):
        original = self.session()
        output = original.save_reconstruction((0,))
        output.write_bytes(b"incomplete")
        repaired = original.save_reconstruction((0,))
        self.assertEqual(read_motion(repaired).frames.shape, original.source.frames.shape)
        with patch("tools.motion_analysis.core.session.fit_pca", wraps=fit_pca) as fit:
            forced = self.session(force=True)
        self.assertEqual(fit.call_count, 1)
        self.assertFalse(forced.fit_cache_hit)
        with patch.object(forced, "reconstruct", wraps=forced.reconstruct) as reconstruct:
            forced.save_reconstruction((0,))
        self.assertEqual(reconstruct.call_count, 1)
        self.assertTrue(self.session().fit_cache_hit)

    def test_native_dof_diagnostics_do_not_modify_motion(self):
        self.features[5, 0] += 9
        write_motion(self.source, self.features)
        source = read_motion(self.source)
        before = source.frames.copy()
        self.assertGreater(discontinuity_diagnostics(source)["possible_branch_jump_count"], 0)
        np.testing.assert_array_equal(source.frames, before)

    def test_invalid_motion_values_and_selection(self):
        for fps in (0, -1, float("nan")):
            write_motion(self.source, self.features, fps=fps)
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.session()
        write_motion(self.source, self.features)
        session = self.session()
        for indices in ((-1,), (session.max_components,), (True,), (1.5,)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                session.reconstruct(indices)

    def test_report_missing_plot_is_regenerated_without_refit(self):
        session = self.session()
        path = session.save_reconstruction((0, 1))
        first = generate_report(session, (0, 1), path)
        self.assertTrue(Path(first["report_path"]).is_file())
        self.assertGreater(first["reconstruction_rmse"], 0)
        Path(first["scores_plot"]).unlink()
        with patch("tools.motion_analysis.core.session.fit_pca", side_effect=AssertionError("Refitted")):
            cached = self.session()
            second = generate_report(cached, (0, 1), path)
        self.assertTrue(Path(second["scores_plot"]).is_file())
        self.assertEqual(json.loads(Path(second["report_path"]).read_text())["selected_components"], [1, 2])


@unittest.skipUnless((ROOT / "data/assets/humanoid/humanoid.xml").is_file()
                     and all((ROOT / f"data/motions/humanoid/humanoid_getup_{name}.pkl").is_file()
                             for name in ("faceup", "facedown")), "Humanoid motion assets unavailable")
class GetupIntegration(unittest.TestCase):
    def test_native_getup_pose_and_window_roundtrip_and_fk(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("faceup", "facedown"):
                source = ROOT / f"data/motions/humanoid/humanoid_getup_{name}.pkl"
                for mode in ("pose", "window"):
                    with self.subTest(name=name, mode=mode):
                        session = load_session(source, mode=mode, cache_root=Path(directory))
                        indices = tuple(range(session.max_components))
                        np.testing.assert_allclose(session.reconstruct(indices), session.source.frames, atol=2e-12)
                        output = session.save_reconstruction(indices)
                        report = generate_report(session, indices, output,
                                                 ROOT / "data/assets/humanoid/humanoid.xml")
                        self.assertLess(report["reconstruction_rmse"], 1e-12)
                        self.assertLess(report["fk_max_cm"], 1e-4)


if __name__ == "__main__":
    unittest.main()
