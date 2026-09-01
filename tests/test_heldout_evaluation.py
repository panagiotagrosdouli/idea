import unittest

import numpy as np

from predictive_pc_fmcw.config import LinkConfig
from predictive_pc_fmcw.learning.heldout import evaluate_checkpoint_arrays
from predictive_pc_fmcw.link import LinkModel


class _ExactPredictor:
    def __init__(self, future):
        self.future = future
        self.offset = 0

    def predict(self, history_xy, horizon_steps, dt_s):
        del horizon_steps, dt_s
        start = self.offset
        self.offset += len(history_xy)
        return self.future[start : self.offset]


class HeldoutEvaluationTest(unittest.TestCase):
    def test_exact_prediction_has_zero_heldout_errors(self):
        history = np.zeros((3, 11, 2), dtype=np.float32)
        future = np.zeros((3, 80, 2), dtype=np.float32)
        future[..., 0] = np.linspace(10, 20, 80)
        headings = np.zeros((3, 80), dtype=np.float32)
        rows = evaluate_checkpoint_arrays(
            predictor=_ExactPredictor(future),
            history_xy=history,
            future_xy=future,
            future_ego_heading_rad=headings,
            scenario_ids=np.asarray(["a", "a", "b"]),
            link_model=LinkModel(LinkConfig()),
            checkpoint="exact.pt",
            objective="trajectory_only",
            seed=1,
            batch_size=2,
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row.ade_m, 0.0)
            self.assertEqual(row.fde_m, 0.0)
            self.assertEqual(row.snr_mae_db, 0.0)
            self.assertEqual(row.goodput_mae_mbps, 0.0)
            self.assertEqual(row.link_lifetime_mae_s, 0.0)


if __name__ == "__main__":
    unittest.main()
