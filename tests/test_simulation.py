import unittest
from dataclasses import replace

import numpy as np

from predictive_pc_fmcw.config import ExperimentConfig
from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.simulation.engine import (
    _current_heading,
    _link_forecast,
    run_simulation,
)
from predictive_pc_fmcw.traffic import generate_traffic_trace


class SimulationTest(unittest.TestCase):
    def test_stationary_heading_reuses_last_valid_direction(self):
        history = np.asarray([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(_current_heading(history), np.pi / 4)

    def test_link_forecast_truncates_at_available_future(self):
        config = ExperimentConfig()
        scenario = generate_synthetic_scenario(4, slots=5, vehicles=2)
        time_index = scenario.timestamps_s.size - 2
        values, _, oracle = _link_forecast(
            scenario,
            time_index,
            10,
            "oracle",
            LinkModel(config.link),
        )
        self.assertEqual(values["snr_db"].shape, (2, 1))
        self.assertTrue(oracle)

    def test_end_to_end_all_schedulers(self):
        base = ExperimentConfig()
        config = replace(base, prediction_horizon_steps=4)
        scenario = generate_synthetic_scenario(
            5, slots=15, vehicles=3, dt_s=config.slot_duration_s
        )
        capacity = LinkModel(config.link).capacity_packets(config.slot_duration_s)
        traffic = generate_traffic_trace(
            7, 15, 3, capacity, config.traffic
        )
        for name in config.benchmark.schedulers:
            result = run_simulation(scenario, name, traffic, config, seed=8)
            self.assertEqual(result.selected_vehicle.shape, (15,))
            self.assertGreaterEqual(result.metrics.packet_delivery_ratio, 0.0)
            self.assertLessEqual(result.metrics.packet_delivery_ratio, 1.0)
            self.assertGreaterEqual(result.metrics.jain_fairness, 0.0)
            self.assertLessEqual(result.metrics.jain_fairness, 1.0)
            self.assertGreaterEqual(
                result.metrics.delivered_before_expiry_ratio, 0.0
            )
            self.assertLessEqual(
                result.metrics.delivered_before_expiry_ratio, 1.0
            )
            self.assertGreaterEqual(
                result.metrics.undelivered_packets_at_disconnect, 0
            )
            self.assertGreaterEqual(result.metrics.p99_latency_ms, 0.0)
            self.assertGreaterEqual(result.metrics.censored_packet_ratio, 0.0)
            self.assertLessEqual(result.metrics.censored_packet_ratio, 1.0)
            accounted = (
                result.metrics.delivered_packets
                + result.metrics.deadline_dropped_packets
                + result.metrics.overflow_dropped_packets
                + result.metrics.remaining_packets
            )
            self.assertEqual(result.metrics.generated_packets, accounted)


if __name__ == "__main__":
    unittest.main()
