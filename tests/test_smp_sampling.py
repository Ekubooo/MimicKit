"""Frozen-prior tracing, native decoding and cache contracts (no physics/RL)."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.motion_analysis.data.motion import read_motion
from tools.motion_analysis.smp.sampling import (
    DEFAULT_CONFIG, DEFAULT_MODEL, SamplingConfig, _load_prior, _sampling_inputs,
    decode_features, export_clip, sample_prior,
)


@unittest.skipUnless(DEFAULT_MODEL.is_file() and DEFAULT_CONFIG.is_file(),
                     "Official frozen spinkick prior is not installed")
class SMPSamplingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.config = SamplingConfig(output_dir=cls.root / "samples", num_samples=2,
                                    batch_size=2, device="cpu")
        torch.manual_seed(819)
        rng_before = torch.random.get_rng_state().clone()
        cls.bundle = sample_prior(cls.config)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
        prior, env, model_path, char_path, _ = _sampling_inputs(cls.config)
        with torch.random.fork_rng():
            cls.model, cls.character, cls.channels = _load_prior(prior, env, model_path, char_path, "cpu")

    def test_trace_callback_preserves_normal_sampling_output_and_rng(self):
        traces = {}

        def trace(step, xt, clean):
            if step in (49, 0):
                traces[step] = (xt.clone(), clean.clone())

        kwargs = dict(shape=(10 * self.channels,), batch_size=2, device="cpu", sampler="ddpm")
        with torch.random.fork_rng():
            torch.manual_seed(0)
            expected_initial = torch.randn(2, 10 * self.channels)
            torch.manual_seed(0)
            ordinary = self.model.sample_ema(**kwargs)
            ordinary_rng = torch.random.get_rng_state().clone()
            torch.manual_seed(0)
            observed = self.model.sample_ema(**kwargs, trace_callback=trace)
            observed_rng = torch.random.get_rng_state().clone()
        torch.testing.assert_close(observed, ordinary, rtol=0, atol=0)
        torch.testing.assert_close(observed_rng, ordinary_rng, rtol=0, atol=0)
        torch.testing.assert_close(traces[49][0], expected_initial, rtol=0, atol=0)
        torch.testing.assert_close(traces[0][1], observed, rtol=0, atol=0)
        expected_features = self.model.unnormalize(ordinary.reshape(2, 10, self.channels)).detach().numpy()
        np.testing.assert_array_equal(self.bundle.features, expected_features)

    def test_shapes_finite_raw_features_and_final_stage(self):
        self.assertEqual(self.bundle.features.shape, (2, 10, 114))
        self.assertEqual(self.bundle.frames.shape, (2, 10, 34))
        self.assertEqual(set(self.bundle.stages), {49, 39, 29, 19, 9, 0})
        self.assertEqual(self.bundle.fps, 30)
        self.assertEqual(self.bundle.loop_mode, 0)
        np.testing.assert_array_equal(self.bundle.stages[0].features, self.bundle.features)
        np.testing.assert_array_equal(self.bundle.stages[0].frames, self.bundle.frames)
        for stage in self.bundle.stages.values():
            self.assertEqual(stage.noise.shape, self.bundle.features.shape)
            self.assertTrue(np.isfinite(stage.noise).all())
            self.assertFalse(stage.noise.flags.writeable)
            self.assertFalse(stage.features.flags.writeable)

    def test_decoder_matches_existing_cpu_exporter_without_projection(self):
        from types import SimpleNamespace
        from tools.diffusion_model.motion_prior_dataset import MotionPriorData

        dataset = MotionPriorData.__new__(MotionPriorData)
        dataset._motion_lib = SimpleNamespace(
            get_num_joints=self.character.get_num_joints,
            get_motion_frame_size=lambda: 6 + self.character.get_dof_size(),
            _kin_char_model=self.character)
        features = torch.tensor(self.bundle.features.copy())
        actual = decode_features(features, self.character)
        expected = torch.stack([dataset.smp_obs_to_motion_frames(clip) for clip in features])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(actual.numpy(), self.bundle.frames)
        self.assertEqual(actual.device, features.device)

    def test_cache_hit_does_not_load_model_or_infer(self):
        with patch("tools.motion_analysis.smp.sampling._load_prior",
                   side_effect=AssertionError("Repeated model inference")):
            cached = sample_prior(self.config)
        self.assertTrue(cached.cache_hit)
        np.testing.assert_array_equal(cached.features, self.bundle.features)
        np.testing.assert_array_equal(cached.stages[19].noise, self.bundle.stages[19].noise)

    def test_export_final_and_stage_are_native_and_reused(self):
        for stage in (None, 19):
            path = export_clip(self.bundle, 1, stage)
            first_mtime = path.stat().st_mtime_ns
            self.assertEqual(export_clip(self.bundle, 1, stage), path)
            self.assertEqual(path.stat().st_mtime_ns, first_mtime)
            motion = read_motion(path)
            expected = self.bundle.frames if stage is None else self.bundle.stages[stage].frames
            np.testing.assert_array_equal(motion.frames, expected[1])
            self.assertEqual(motion.fps, 30)
            self.assertEqual(motion.loop_mode, 0)
        for index, stage in ((-1, None), (2, None), (0, 100), (0, 1.5)):
            with self.subTest(index=index, stage=stage), self.assertRaises(ValueError):
                export_clip(self.bundle, index, stage)

    def test_parameter_and_source_hash_changes_invalidate_cache(self):
        sentinel = object()
        changes = ({"seed": 1}, {"batch_size": 1}, {"num_samples": 1},
                   {"use_ema": False}, {"trace_steps": ()}, {"version": "v2"})
        for change in changes:
            with self.subTest(change=change), patch("tools.motion_analysis.smp.sampling._generate", return_value=sentinel) as generate:
                self.assertIs(sample_prior(replace(self.config, **change)), sentinel)
                self.assertEqual(generate.call_count, 1)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "prior.yaml"
            config_path.write_bytes(DEFAULT_CONFIG.read_bytes() + b"\n# content changed\n")
            with patch("tools.motion_analysis.smp.sampling._generate", return_value=sentinel) as generate:
                self.assertIs(sample_prior(replace(self.config, config_path=config_path)), sentinel)
                self.assertEqual(generate.call_count, 1)

    def test_corrupt_cache_and_force_request_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            destination = output / self.bundle.cache_dir.name
            shutil.copytree(self.bundle.cache_dir, destination)
            configuration = replace(self.config, output_dir=output)
            (destination / "samples.npz").write_bytes(b"incomplete")
            with patch("tools.motion_analysis.smp.sampling._generate", return_value=self.bundle) as generate:
                sample_prior(configuration)
                self.assertEqual(generate.call_count, 1)
            shutil.copyfile(self.bundle.cache_dir / "samples.npz", destination / "samples.npz")
            (destination / "manifest.json").write_text("[]", encoding="utf-8")
            with patch("tools.motion_analysis.smp.sampling._generate", return_value=self.bundle) as generate:
                sample_prior(configuration)
                self.assertEqual(generate.call_count, 1)
        with patch("tools.motion_analysis.smp.sampling._generate", return_value=self.bundle) as generate:
            sample_prior(replace(self.config, force=True))
            self.assertEqual(generate.call_count, 1)

    def test_invalid_config_and_decoder_shapes_fail_before_sampling(self):
        for change in ({"num_samples": 0}, {"batch_size": 0}, {"seed": -1},
                       {"trace_steps": (50,)}, {"trace_steps": (-1,)}, {"version": "../bad"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                sample_prior(replace(self.config, **change))
        for features in (torch.zeros(10, 114), torch.zeros(1, 10, 8), torch.full((1, 10, 114), float("nan"))):
            with self.subTest(shape=features.shape), self.assertRaises(ValueError):
                decode_features(features, self.character)


if __name__ == "__main__":
    unittest.main()
