import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.data.womd_preflight import preflight_womd_roots


class WomdPreflightTest(unittest.TestCase):
    def _write_corpus(self, path: Path, scenario_ids: list[str], splits: list[str]):
        samples = len(scenario_ids)
        np.savez_compressed(
            path,
            history_xy=np.zeros((samples, 11, 2)),
            future_xy=np.ones((samples, 80, 2)),
            scenario_id=np.asarray(scenario_ids),
            actor_id=np.asarray([f"actor-{index}" for index in range(samples)]),
            future_ego_heading_rad=np.zeros((samples, 80)),
            split=np.asarray(splits),
            source=np.asarray("real_WOMD_v1.3.1_true_SDC_geometry"),
            coordinate_frame=np.asarray("world_xy_with_explicit_ego_heading"),
        )

    def test_empty_root_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = preflight_womd_roots([temporary])
        self.assertEqual(report["status"], "BLOCKED_NO_DATA")
        self.assertFalse(report["stage1_ready"])

    def test_arbitrary_npz_does_not_imply_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root / "proxy.npz", positions=np.zeros((2, 2)))
            report = preflight_womd_roots([root])
        self.assertEqual(report["status"], "BLOCKED_INVALID_CORPUS")
        self.assertFalse(report["npz_candidates"][0]["valid_npz_schema"])

    def test_tfrecord_requires_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.tfrecord-00000-of-00001"
            path.write_bytes(b"candidate")
            report = preflight_womd_roots([path])
        self.assertEqual(report["status"], "BLOCKED_EXPORT_REQUIRED")

    def test_only_verified_disjoint_pair_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_corpus(
                root / "training.npz", ["a", "b"], ["training", "development"]
            )
            self._write_corpus(
                root / "validation.npz",
                ["c", "d"],
                ["official_validation", "official_validation"],
            )
            report = preflight_womd_roots([root])
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["stage1_ready"])
        self.assertEqual(report["counts"]["pairs_checked"], 1)

    def test_scenario_overlap_blocks_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_corpus(
                root / "training.npz",
                ["shared", "b"],
                ["training", "development"],
            )
            self._write_corpus(
                root / "validation.npz",
                ["shared", "d"],
                ["official_validation", "official_validation"],
            )
            report = preflight_womd_roots([root])
        self.assertEqual(report["status"], "BLOCKED_INVALID_CORPUS")
        verification = report["pair_checks"][0]["verification"]
        self.assertEqual(verification["cross_corpus_overlap"]["count"], 1)


if __name__ == "__main__":
    unittest.main()
