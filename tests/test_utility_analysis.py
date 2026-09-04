import csv
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.learning.utility_analysis import (
    UTILITY_METRICS,
    aggregate_scenario_relationship,
    join_accuracy_and_scheduler_utility,
    summarize_utility,
)


def utility_row(
    scenario_id: str,
    seed: int,
    ade_m: float,
    goodput_gain: float,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "objective": "a",
        "seed": seed,
        "scenario_id": scenario_id,
        "ade_m": ade_m,
        "fde_m": ade_m,
    }
    for metric in UTILITY_METRICS:
        baseline = 1.0
        proposed = baseline + (goodput_gain if metric == "goodput_mbps" else 0.1)
        row[f"learned_{metric}"] = proposed
        row[f"reactive_{metric}"] = baseline
        row[f"delta_{metric}"] = proposed - baseline
    return row


class UtilityAnalysisTest(unittest.TestCase):
    def test_join_uses_same_objective_seed_and_scenario(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            heldout = root / "heldout.csv"
            with heldout.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "objective",
                        "seed",
                        "scenario_id",
                        "ade_m",
                        "fde_m",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "objective": "full",
                        "seed": 1,
                        "scenario_id": "s",
                        "ade_m": 1,
                        "fde_m": 2,
                    }
                )
            run = root / "scheduler" / "full" / "seed_1"
            run.mkdir(parents=True)
            fields = [
                "scheduler",
                "scenario_id",
                "goodput_mbps",
                "packet_delivery_ratio",
                "scheduled_outage_fraction",
                "p95_latency_ms",
                "deadline_miss_ratio",
                "jain_fairness",
            ]
            metrics_path = run / "episode_metrics.csv"
            with metrics_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                base = {
                    "scenario_id": "s",
                    "packet_delivery_ratio": 0.8,
                    "scheduled_outage_fraction": 0.2,
                    "p95_latency_ms": 10,
                    "deadline_miss_ratio": 0.1,
                    "jain_fairness": 0.9,
                }
                writer.writerow(
                    {
                        **base,
                        "scheduler": "reactive_greedy",
                        "goodput_mbps": 1.0,
                    }
                )
                writer.writerow(
                    {
                        **base,
                        "scheduler": "learned_predictive",
                        "goodput_mbps": 1.2,
                    }
                )
            rows = join_accuracy_and_scheduler_utility(heldout, root / "scheduler")
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(rows[0]["delta_goodput_mbps"], 0.2)
            summary = summarize_utility(rows)
            self.assertEqual(summary["full"]["independent_scenarios"], 1)

    def test_relationship_collapses_model_seeds_within_scenario(self):
        rows = [
            utility_row("s1", 1, 1.0, 0.1),
            utility_row("s1", 2, 3.0, 0.3),
            utility_row("s2", 1, 4.0, -0.2),
        ]
        aggregated = aggregate_scenario_relationship(rows)
        self.assertEqual(len(aggregated), 2)
        first = aggregated[0]
        self.assertEqual(first["scenario_id"], "s1")
        self.assertAlmostEqual(first["ade_m"], 2.0)
        self.assertAlmostEqual(first["delta_goodput_mbps"], 0.2)
        self.assertEqual(first["model_seed_rows"], 2)

        summary = summarize_utility(rows)
        relationship = summary["ade_vs_realized_goodput_gain"]
        self.assertEqual(relationship["scenario_seed_rows"], 3)
        self.assertEqual(relationship["independent_scenarios"], 2)
        self.assertIn("averaged", relationship["aggregation"])


if __name__ == "__main__":
    unittest.main()
