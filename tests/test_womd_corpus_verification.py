import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import _01_verify_womd_corpora  # type: ignore[attr-defined]


class WomdCorpusVerificationTest(unittest.TestCase):
    def _write(self, path: Path, scenario_ids: list[str], split: str) -> None:
        samples = len(scenario_ids)
        np.savez_compressed(
            path,
            history_xy=np.zeros((samples, 11, 2)),
            future_xy=np.ones((samples, 80, 2)),
            scenario_id=np.asarray(scenario_ids),
            actor_id=np.asarray([str(index) for index in range(samples)]),
            future_ego_heading_rad=np.zeros((samples, 80)),
            split=np.asarray([split] * samples),
            source=np.asarray("real_WOMD_v1.3.1_true_SDC_geometry"),
            coordinate_frame=np.asarray("world_xy_with_explicit_ego_heading"),
        )

    def test_zero_overlap_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.npz"
            validation = root / "validation.npz"
            self._write(training, ["a", "b"], "training")
            self._write(validation, ["c", "d"], "validation")
            report = _01_verify_womd_corpora.verify_corpora(training, validation)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["cross_corpus_overlap"]["count"], 0)

    def test_overlap_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.npz"
            validation = root / "validation.npz"
            self._write(training, ["a", "shared"], "training")
            self._write(validation, ["shared", "z"], "validation")
            report = _01_verify_womd_corpora.verify_corpora(training, validation)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["cross_corpus_overlap"]["scenario_ids"], ["shared"])


if __name__ == "__main__":
    unittest.main()
