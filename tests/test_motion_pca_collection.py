"""Numerical and saved-analysis contracts for multi-clip PCA collections."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.motion_analysis.core.collection import (
    fit_collection, load_collection, project_collection, save_collection,
)
from tools.motion_analysis.data.motion import MotionData
from tools.motion_analysis.core.samples import build_samples


def make_clip(name, features, *, fps=30.0, loop=0, dtype=np.float64):
    features = np.asarray(features)
    frames = np.zeros((len(features), features.shape[1] + 6), dtype=dtype)
    frames[:, 0] = np.arange(len(features))
    frames[:, 2] = 1
    frames[:, 6:] = features
    return MotionData(Path(name), frames, fps, loop,
                      hashlib.sha256(frames.tobytes()).hexdigest())


class CollectionNumerics(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(73)
        self.clips = (make_clip("one.pkl", rng.normal(size=(17, 3)), fps=60, loop=1),
                      make_clip("two.pkl", rng.normal(size=(14, 3)) + 2, fps=30, loop=0))

    def test_pose_and_window_fitted_data_reconstruct_at_full_rank(self):
        for mode in ("pose", "window"):
            with self.subTest(mode=mode):
                analysis = fit_collection(self.clips, mode=mode, window=5, stride=3)
                indices = tuple(range(analysis.max_components))
                for i, source in enumerate(self.clips):
                    session = analysis.for_clip(i)
                    np.testing.assert_allclose(session.reconstruct(indices), source.frames, atol=1e-12)
                    np.testing.assert_array_equal(session.reconstruct(())[:, :6], source.frames[:, :6])
                    self.assertEqual(session.source.fps, source.fps)
                    self.assertEqual(session.source.loop_mode, source.loop_mode)
                    self.assertIs(session.model, analysis.model)
                self.assertTrue(analysis.fitted_on_collection)
                self.assertEqual(analysis.model.sample_count, len(analysis.samples))

    def test_one_window_per_clip_is_valid_but_single_file_default_is_unchanged(self):
        clips = (make_clip("a.pkl", np.arange(30).reshape(10, 3)),
                 make_clip("b.pkl", np.arange(30, 60).reshape(10, 3)))
        with self.assertRaisesRegex(ValueError, "at least two windows"):
            build_samples(clips[0].frames[:, 6:], "window", 10)
        analysis = fit_collection(clips, mode="window", window=10)
        self.assertEqual(analysis.samples.shape, (2, 30))
        for index in range(2):
            self.assertEqual(analysis.for_clip(index).scores.shape, (1, 1))
            np.testing.assert_array_equal(analysis.layouts[index].starts, [0])
        np.testing.assert_array_equal(analysis.samples[0], clips[0].frames[:, 6:].reshape(-1))
        np.testing.assert_array_equal(analysis.samples[1], clips[1].frames[:, 6:].reshape(-1))
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            fit_collection(clips[:1], mode="window", window=10)

    def test_shared_unseen_data_can_have_full_rank_projection_residual(self):
        time = np.arange(16, dtype=float)
        reference = make_clip("reference.pkl", np.column_stack((time, time * 2, time * 0)))
        reference_fit = fit_collection([reference], mode="window", window=10)
        unseen = make_clip("new.pkl", np.column_stack((time[:10], time[:10] * 2, np.ones(10))))
        projected = project_collection([unseen], reference_fit.model, mode="window", window=10)
        self.assertIs(projected.model, reference_fit.model)
        self.assertFalse(projected.fitted_on_collection)
        self.assertEqual(projected.samples.shape, (1, 30))
        result = projected.for_clip(0).reconstruct(tuple(range(projected.max_components)))
        self.assertGreater(float(np.mean((result[:, 6:] - unseen.frames[:, 6:]) ** 2)), 0.1)
        np.testing.assert_array_equal(result[:, :6], unseen.frames[:, :6])

    def test_external_integer_and_float32_motion_frames_are_normalized(self):
        for dtype in (np.int32, np.float32):
            with self.subTest(dtype=dtype):
                original = make_clip("typed.pkl", [[0, 0], [1, 1], [1, 2], [2, 3]], dtype=dtype)
                analysis = fit_collection([original])
                session = analysis.for_clip(0)
                self.assertEqual(session.source.frames.dtype, np.float64)
                self.assertEqual(original.frames.dtype, dtype)
                mean_only = session.reconstruct(())
                np.testing.assert_allclose(mean_only[:, 6:], [[1, 1.5]] * 4, atol=1e-14)
                np.testing.assert_allclose(
                    session.reconstruct(tuple(range(analysis.max_components))),
                    original.frames, atol=1e-13)

    def test_constant_collection_has_mean_only_reconstruction(self):
        clips = [make_clip(f"clip_{i}.pkl", np.full((10, 3), 0.1)) for i in range(3)]
        analysis = fit_collection(clips, mode="window", window=10)
        self.assertEqual(analysis.max_components, 0)
        for i, clip in enumerate(clips):
            np.testing.assert_array_equal(analysis.for_clip(i).reconstruct(()), clip.frames)

    def test_collection_rejects_empty_mismatched_and_invalid_clips(self):
        with self.assertRaises(ValueError):
            fit_collection([])
        with self.assertRaisesRegex(ValueError, "same pose feature"):
            fit_collection([self.clips[0], make_clip("bad.pkl", np.ones((10, 2)))])
        bad = self.clips[0].frames.copy()
        bad[2, 7] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_collection([MotionData(Path("bad.pkl"), bad, 30, 0, "bad")])
        with self.assertRaises(ValueError):
            fit_collection([self.clips[0]], mode="window", window=50)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            project_collection([self.clips[0]], fit_collection(self.clips).model,
                               mode="window", window=5)
        for index in (-1, 2, True, 0.5):
            with self.assertRaises(ValueError):
                fit_collection(self.clips).for_clip(index)


class SavedCollectionContracts(unittest.TestCase):
    def setUp(self):
        (ROOT / "output").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "output")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "analysis.json"
        rng = np.random.default_rng(18)
        self.clips = tuple(make_clip(f"absent_{i}.pkl", rng.normal(size=(10, 3))) for i in range(4))
        self.analysis = fit_collection(self.clips, mode="window", window=10)
        self.entry = save_collection(self.analysis, self.manifest, group="generated",
                                     source_ids=[f"sample_{i:03d}" for i in range(4)])

    def test_snapshot_load_never_fits_and_preserves_source_and_projections(self):
        with patch("tools.motion_analysis.core.collection.fit_pca", side_effect=AssertionError("must not fit")), \
                patch("numpy.linalg.svd", side_effect=AssertionError("must not SVD")):
            loaded = load_collection(self.manifest, group="generated", mode="window")
            self.assertTrue(loaded.fit_cache_hit)
            self.assertEqual(loaded.source_ids[2], "sample_002")
            self.assertEqual(loaded.samples.shape, (4, 30))
            for i, source in enumerate(self.clips):
                self.assertFalse(source.path.exists())
                np.testing.assert_array_equal(loaded.clips[i].frames, source.frames)
                np.testing.assert_array_equal(loaded.scores[i], self.analysis.scores[i])
                session = loaded.for_clip(i)
                self.assertEqual(session.source_id, f"sample_{i:03d}")
                self.assertEqual(session.layout.window, 10)
                np.testing.assert_allclose(session.reconstruct(tuple(range(loaded.max_components))),
                                           source.frames, atol=1e-12)

    def test_models_and_motion_frames_are_deduplicated_across_entries(self):
        save_collection(self.analysis, self.manifest, group="generated", basis="independent")
        shared = project_collection(self.clips, self.analysis.model, mode="window", window=10)
        save_collection(shared, self.manifest, group="stage_49", basis="shared")
        pose = fit_collection(self.clips)
        save_collection(pose, self.manifest, group="generated", basis="shared")
        index = json.loads(self.manifest.read_text())
        self.assertEqual(len(index["entries"]), 4)
        self.assertEqual(len(list((self.root / "models").glob("*.npz"))), 2)
        self.assertEqual(len(list((self.root / "motions").glob("*.npz"))), 1)
        self.assertFalse(load_collection(self.manifest, group="stage_49", mode="window").fitted_on_collection)
        save_collection(pose, self.manifest, group="generated", basis="shared")
        self.assertEqual(len(json.loads(self.manifest.read_text())["entries"]), 4)

    def test_projection_snapshot_loads_with_one_window_and_reference_sample_count(self):
        projection = project_collection([self.clips[0]], self.analysis.model, mode="window", window=10)
        save_collection(projection, self.manifest, group="one")
        loaded = load_collection(self.manifest, group="one", mode="window")
        self.assertEqual(loaded.model.sample_count, 4)
        self.assertEqual(len(loaded.samples), 1)
        self.assertEqual(loaded.for_clip(0).scores.shape, (1, loaded.max_components))

    def test_manifest_records_playback_settings_and_save_is_explicit(self):
        save_collection(self.analysis, self.manifest, group="generated",
                        env_config=ROOT / "data/envs/view_motion_humanoid_env.yaml")
        loaded = load_collection(self.manifest, group="generated", mode="window")
        session = loaded.for_clip(1)
        before = set(self.root.rglob("*"))
        session.reconstruct(())
        self.assertEqual(set(self.root.rglob("*")), before)
        path = session.save_reconstruction(())
        self.assertTrue(path.is_file())
        import pickle
        saved = pickle.loads(path.read_bytes())
        self.assertEqual(saved["fps"], self.clips[1].fps)
        self.assertEqual(saved["loop_mode"], self.clips[1].loop_mode)
        np.testing.assert_array_equal(np.asarray(saved["frames"])[:, :6], self.clips[1].frames[:, :6])
        index = json.loads(self.manifest.read_text())
        self.assertEqual(index["playback"]["env_config"],
                         str((ROOT / "data/envs/view_motion_humanoid_env.yaml").resolve()))

    def test_corrupt_or_missing_archive_fails_without_fallback(self):
        artifact = self.root / self.entry["data"]["path"]
        original = artifact.read_bytes()
        artifact.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_collection(self.manifest, group="generated", mode="window")
        artifact.write_bytes(original)
        artifact.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            load_collection(self.manifest, group="generated", mode="window")

    def test_invalid_numeric_snapshot_is_rejected_even_with_updated_archive_hash(self):
        index = json.loads(self.manifest.read_text())
        descriptor = index["entries"][0]["motions"]
        artifact = self.root / descriptor["path"]
        with np.load(artifact, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["frames"][0, 6] = np.nan
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **arrays)
        artifact.write_bytes(buffer.getvalue())
        descriptor["sha256"] = hashlib.sha256(buffer.getvalue()).hexdigest()
        self.manifest.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_collection(self.manifest, group="generated", mode="window")

    def test_source_metadata_and_path_escape_validation(self):
        index = json.loads(self.manifest.read_text())
        index["entries"][0]["clips"][0]["shape"] = [999, 99]
        self.manifest.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "metadata"):
            load_collection(self.manifest, group="generated", mode="window")
        index["entries"][0]["model"]["path"] = "../outside.npz"
        self.manifest.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "inside"):
            load_collection(self.manifest, group="generated", mode="window")

    def test_index_lookup_and_identity_errors(self):
        with self.assertRaisesRegex(ValueError, "available"):
            load_collection(self.manifest, group="missing", mode="window")
        with self.assertRaisesRegex(ValueError, "source_ids"):
            save_collection(self.analysis, self.manifest, group="generated", source_ids=["same"] * 4)
        with self.assertRaisesRegex(ValueError, "Group"):
            save_collection(self.analysis, self.manifest, group="../escape")
        index = json.loads(self.manifest.read_text())
        index["complete"] = False
        self.manifest.write_text(json.dumps(index))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            load_collection(self.manifest, group="generated", mode="window")


if __name__ == "__main__":
    unittest.main()
