import unittest

import numpy as np

from predictive_pc_fmcw.config import SchedulerConfig, TrafficConfig
from predictive_pc_fmcw.scheduling.base import SchedulerContext
from predictive_pc_fmcw.scheduling.policies import build_scheduler
from predictive_pc_fmcw.traffic import generate_traffic_trace


class TrafficAndSchedulerTest(unittest.TestCase):
    def test_traffic_reproducibility(self):
        first = generate_traffic_trace(4, 8, 3, 20, TrafficConfig())
        second = generate_traffic_trace(4, 8, 3, 20, TrafficConfig())
        np.testing.assert_array_equal(first.arrivals, second.arrivals)
        np.testing.assert_array_equal(first.success_uniforms, second.success_uniforms)
        self.assertEqual(first.deadlines, second.deadlines)

    def test_all_policies_choose_at_most_one_eligible_vehicle(self):
        context = SchedulerContext(
            slot=0,
            queue_lengths=np.asarray([0, 3, 2]),
            time_to_deadline=np.asarray([np.inf, 2.0, 5.0]),
            current_goodput_bps=np.asarray([1e9, 8e8, 7e8]),
            current_outage=np.asarray([False, False, False]),
            predicted_goodput_bps=np.full((3, 4), 8e8),
            predicted_outage=np.zeros((3, 4), dtype=bool),
            predicted_lifetime_steps=np.asarray([4, 2, 4]),
            delivered_bits=np.asarray([0.0, 10.0, 4.0]),
            previous_vehicle=None,
            data_rate_bps=1e9,
            discount=0.9,
            oracle_forecast=True,
        )
        names = [
            "random",
            "round_robin",
            "reactive_greedy",
            "proportional_fair",
            "cv_predictive",
            "predictive_utility",
            "link_lifetime",
            "oracle",
        ]
        for name in names:
            decision = build_scheduler(name, SchedulerConfig(), 9).select(context)
            self.assertIn(decision.vehicle, {1, 2})


if __name__ == "__main__":
    unittest.main()

