import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.data.corpus_audit import verify_corpora


class WomdCorpusVerificationTest(unittest.TestCase):
    def _write(
        self,
        path: Path,
        scenario_ids: list[str],
        splits: list[str],
        *,
        source: str = "real_WOMD_v1.3.1_true_SDC_geometry",
        coordinate_frame: str = "world_xy_with_explicit_ego_heading",
    ) -> None:
        samples = len(scenario_ids)
        if len(splits) != samples:
            raise ValueError("splits must align with scenario_ids")
        np.savez_compressed(
            path,
            history_xy=np.zeros((samples, 11, 2)),
            future_xy=np.ones((samples, 80, 2)),
            scenario_id=np.asarray(scenario_ids),
            actor_id=np.asarray([str(index) for index in range(samples)]),
            future_ego_heading_rad=np.zeros((samples, 80)),
            split=np.asarray(splits),
            source=np.asarray(source),
            coordinate_frame=np.asarray(coordinate_frame),
        )

    def _valid_pair(self, root: Path) -> tuple[Path, Path]:
        training = root / "training.npz"
        validation = root / "validation.npz"
        self._write(training, ["a", "b"], ["training", "development"])
        self._write(validation, ["c", "d"], ["official_validation"] * 2)
        return training, validation

    def test_zero_overlap_and_provenance_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            training, validation = self._valid_pair(Path(temporary))
            report = verify_corpora(training, validation)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["cross_corpus_overlap"]["count"], 0)
            self.assertTrue(all(report["provenance_checks"].values()))

    def test_overlap_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training.npz"
            validation = root / "validation.npz"
            self._write(training, ["a", "shared"], ["training", "development"])
            self._write(validation, ["shared", "z"], ["official_validation"] * 2)
            report = verify_corpora(training, validation)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["cross_corpus_overlap"]["scenario_ids"], ["shared"])

    def test_proxy_geometry_source_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, validation = self._valid_pair(root)
            self._write(
                validation,
                ["c", "d"],
                ["official_validation"] * 2,
                source="real_WOMD_motion_proxy_ego_geometry",
            )
            report = verify_corpora(training, validation)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(
                report["provenance_checks"]["validation_official_true_sdc_source"]
            )

    def test_validation_role_cannot_be_development(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training, validation = self._valid_pair(root)
            self._write(validation, ["c", "d"], ["development"] * 2)
            report = verify_corpora(training, validation)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["provenance_checks"]["validation_is_untouched_role"])


if __name__ == "__main__":
    unittest.main()
