import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.data.womd_export import load_womd_motion_scenarios
from predictive_pc_fmcw.predictors import ConstantVelocityPredictor, forecast_scenario


class DataAndPredictorTest(unittest.TestCase):
    def test_synthetic_shapes(self):
        scenario = generate_synthetic_scenario(1, slots=12, vehicles=3)
        self.assertEqual(scenario.vehicle_positions_xy.shape, (22, 3, 2))
        self.assertEqual(scenario.evaluation_slots, 12)

    def test_future_mutation_does_not_change_causal_prediction(self):
        scenario = generate_synthetic_scenario(2, slots=12, vehicles=2)
        combined = scenario.combined_positions()
        changed = combined.copy()
        changed[scenario.start_index + 1 :] += 1e5
        first = forecast_scenario(
            combined,
            scenario.start_index,
            5,
            scenario.dt_s,
            ConstantVelocityPredictor(),
        )
        second = forecast_scenario(
            changed,
            scenario.start_index,
            5,
            scenario.dt_s,
            ConstantVelocityPredictor(),
        )
        np.testing.assert_array_equal(first.vehicle_xy, second.vehicle_xy)

    def test_womd_proxy_adapter(self):
        records = []
        for actor in range(3):
            records.append(
                {
                    "scenario_id": "scene",
                    "track_index": actor,
                    "past": [[float(t + actor * 4), 0.0] for t in range(3)],
                    "future": [[float(t + 3 + actor * 4), 0.0] for t in range(2)],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.json"
            path.write_text(json.dumps(records), encoding="utf-8")
            scenarios = load_womd_motion_scenarios(path)
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].vehicle_count, 2)
        self.assertIn("proxy", scenarios[0].source)


if __name__ == "__main__":
    unittest.main()

