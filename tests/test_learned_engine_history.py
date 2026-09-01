import unittest

import numpy as np

from predictive_pc_fmcw.config import LinkConfig
from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.simulation.engine import _link_forecast


class _HistoryCheckingPredictor:
    feature_schema = {"history_steps": 3}

    def __init__(self):
        self.received_history_steps = []

    def predict(self, history_xy, horizon_steps, dt_s):
        del dt_s
        self.received_history_steps.append(history_xy.shape[1])
        return np.repeat(history_xy[:, -1:, :], horizon_steps, axis=1)


class LearnedEngineHistoryTest(unittest.TestCase):
    def test_learned_forecast_uses_checkpoint_history_window(self):
        scenario = generate_synthetic_scenario(seed=5, slots=20, vehicles=2)
        predictor = _HistoryCheckingPredictor()
        _link_forecast(
            scenario,
            time_index=8,
            horizon=5,
            mode="learned",
            model=LinkModel(LinkConfig()),
            learned_predictor=predictor,
        )
        self.assertEqual(predictor.received_history_steps, [3])


if __name__ == "__main__":
    unittest.main()
