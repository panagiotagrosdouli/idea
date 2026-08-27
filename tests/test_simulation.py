import unittest
from dataclasses import replace

from predictive_pc_fmcw.config import ExperimentConfig
from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.simulation.engine import run_simulation
from predictive_pc_fmcw.traffic import generate_traffic_trace


class SimulationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

