import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.data.dataset_audit import audit_training_npz


class DatasetAuditTest(unittest.TestCase):
    def test_audit_reports_shapes_splits_and_finiteness(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.npz"
            np.savez_compressed(
                path,
                history_xy=np.zeros((2, 11, 2)),
                future_xy=np.ones((2, 80, 2)),
                scenario_id=np.asarray(["a", "b"]),
                actor_id=np.asarray(["1", "2"]),
                future_ego_heading_rad=np.zeros((2, 80)),
                split=np.asarray(["training", "development"]),
                source=np.asarray("real_WOMD_v1.3.1_true_SDC_geometry"),
                coordinate_frame=np.asarray("world_xy_with_explicit_ego_heading"),
            )
            report = audit_training_npz(path)
            self.assertEqual(report["sample_count"], 2)
            self.assertTrue(report["scenario_split_integrity"]["passed"])
            self.assertEqual(report["unique_scenarios"], 2)
            self.assertEqual(report["splits"]["training"], 1)
            self.assertTrue(report["all_numeric_values_finite"])
            self.assertEqual(len(report["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
