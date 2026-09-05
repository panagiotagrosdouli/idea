import unittest

from predictive_pc_fmcw.learning.analysis import (
    accuracy_link_correlations,
    paired_full_vs_trajectory,
    summarize_objectives,
)


def _row(objective, seed, scenario, offset):
    return {
        "objective": objective,
        "seed": seed,
        "scenario_id": scenario,
        "ade_m": 1.0 + offset,
        "fde_m": 2.0 + offset,
        "range_mae_m": 0.5 + offset,
        "bearing_mae_deg": 3.0 + offset,
        "snr_mae_db": 4.0 + offset,
        "goodput_mae_mbps": 0.2 + offset,
        "outage_f1": 0.8 - offset,
        "outage_auroc": 0.9 - offset,
        "link_lifetime_mae_s": 0.4 + offset,
    }


class LearnedAnalysisTest(unittest.TestCase):
    def test_summary_and_paired_comparison_are_scenario_aligned(self):
        rows = []
        for seed in (1, 2, 3):
            for scenario in ("a", "b"):
                rows.append(_row("trajectory_only", seed, scenario, 0.2))
                rows.append(_row("full", seed, scenario, 0.0))
        summary = summarize_objectives(rows)
        self.assertEqual(summary["full"]["scenarios"], 2)
        self.assertEqual(summary["full"]["seeds"], 3)
        paired = paired_full_vs_trajectory(rows)
        self.assertEqual(paired["pairs"], 6)
        self.assertGreater(
            paired["metrics"]["ade_m"]["favorable_mean_difference"], 0
        )

    def test_accuracy_link_correlation_uses_one_row_per_scenario(self):
        rows = [
            _row("trajectory_only", 1, "a", 0.0),
            _row("full", 1, "a", 0.2),
            _row("trajectory_only", 2, "a", 0.4),
            _row("full", 1, "b", 1.0),
            _row("trajectory_only", 2, "b", 1.2),
        ]
        correlation = accuracy_link_correlations(rows)
        self.assertEqual(correlation["model_seed_rows"], 5)
        self.assertEqual(correlation["independent_scenarios"], 2)
        self.assertIn("averaged", correlation["aggregation"])


if __name__ == "__main__":
    unittest.main()
