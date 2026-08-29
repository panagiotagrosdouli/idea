import unittest

import numpy as np

from predictive_pc_fmcw.config import SensingConfig
from predictive_pc_fmcw.sensing import observe_combined_history


class SensingTest(unittest.TestCase):
    def setUp(self):
        time = np.arange(6, dtype=float)
        ego = np.stack([time, np.zeros_like(time)], axis=-1)
        near = ego + np.asarray([20.0, 2.0])
        far = ego + np.asarray([100.0, -4.0])
        self.positions = np.stack([ego, near, far], axis=1)

    def test_perfect_observation_is_identity(self):
        result = observe_combined_history(
            self.positions, SensingConfig(model="perfect"), seed=3
        )
        np.testing.assert_array_equal(result.positions_xy, self.positions)
        self.assertFalse(result.measured_data)

    def test_range_bearing_noise_is_reproducible_and_keeps_ego_exact(self):
        config = SensingConfig(model="range_bearing_assumed")
        first = observe_combined_history(self.positions, config, seed=9)
        second = observe_combined_history(self.positions, config, seed=9)
        np.testing.assert_allclose(first.positions_xy, second.positions_xy)
        np.testing.assert_array_equal(first.positions_xy[:, 0], self.positions[:, 0])
        self.assertGreater(
            first.position_std_m[:, 2].mean(),
            first.position_std_m[:, 1].mean(),
        )

    def test_temporal_correlation_validation(self):
        with self.assertRaises(ValueError):
            SensingConfig(temporal_correlation=1.0)


if __name__ == "__main__":
    unittest.main()
