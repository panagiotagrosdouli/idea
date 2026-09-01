import csv
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.learning.utility_analysis import (
    join_accuracy_and_scheduler_utility,
    summarize_utility,
)


class UtilityAnalysisTest(unittest.TestCase):
    def test_join_uses_same_objective_seed_and_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout = root / "heldout.csv"
            with heldout.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["objective", "seed", "scenario_id", "ade_m", "fde_m"],
                )
                writer.writeheader()
                writer.writerow({"objective": "full", "seed": 1, "scenario_id": "s", "ade_m": 1, "fde_m": 2})
            run = root / "scheduler" / "full" / "seed_1"
            run.mkdir(parents=True)
            fields = ["scheduler", "scenario_id", *(
                "goodput_mbps", "packet_delivery_ratio", "scheduled_outage_fraction",
                "p95_latency_ms", "deadline_miss_ratio", "jain_fairness",
            )]
            with (run / "episode_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                base = {"scenario_id": "s", "packet_delivery_ratio": 0.8, "scheduled_outage_fraction": 0.2, "p95_latency_ms": 10, "deadline_miss_ratio": 0.1, "jain_fairness": 0.9}
                writer.writerow({**base, "scheduler": "reactive_greedy", "goodput_mbps": 1.0})
                writer.writerow({**base, "scheduler": "learned_predictive", "goodput_mbps": 1.2})
            rows = join_accuracy_and_scheduler_utility(heldout, root / "scheduler")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["delta_goodput_mbps"], 0.2)
            summary = summarize_utility(rows)
            self.assertEqual(summary["full"]["independent_scenarios"], 1)


if __name__ == "__main__":
    unittest.main()
