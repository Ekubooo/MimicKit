"""Case-level cache recovery, without model inference or a viewer."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.motion_analysis.workflows.spinkick import prepare_case
from tools.motion_analysis.smp.sampling import SampleBundle, StageSamples


class CaseCacheContracts(unittest.TestCase):
    def test_complete_cache_avoids_svd_and_corrupt_index_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            case = Path(directory)
            data = case / "assets/data"
            data.mkdir(parents=True)
            prior_manifest = data / "fake_prior.json"
            prior_manifest.write_text('{"test_fixture": true}', encoding="utf-8")
            rng = np.random.default_rng(7)
            frames = rng.normal(scale=.1, size=(2, 10, 34))
            frames[:, :, 2] = 1
            features = np.zeros((2, 10, 114))
            bundle = SampleBundle(prior_manifest, True, 30., 0, features, frames,
                                  {0: StageSamples(features, features, frames)})
            with patch("tools.motion_analysis.workflows.spinkick.sample_prior", return_value=bundle):
                path = prepare_case(case, num_samples=2, device="cpu")
                with patch("numpy.linalg.svd", side_effect=AssertionError("Repeated fit")):
                    self.assertEqual(prepare_case(case, num_samples=2, device="cpu"), path)
                path.write_text("{broken JSON", encoding="utf-8")
                prepare_case(case, num_samples=2, device="cpu")
                index = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(index["entries"]), 10)
                with patch("numpy.linalg.svd", side_effect=AssertionError("Repeated fit")):
                    prepare_case(case, num_samples=2, device="cpu")


if __name__ == "__main__":
    unittest.main()
