import unittest
import warnings

import numpy as np

from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.metrics import (
    holm_adjusted_pvalues,
    paired_metric_statistics,
)
from predictive_pc_fmcw.scenario_slices import classify_scenario_slices


class MetricsAndSlicesTest(unittest.TestCase):
    def test_holm_adjustment_is_monotone_and_bounded(self):
        adjusted = holm_adjusted_pvalues([0.01, 0.04, 0.03])
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))
        self.assertGreaterEqual(adjusted[0], 0.01)

    def test_clustered_paired_statistics_use_independent_seeds(self):
        proposed = [2.0, 2.2, 0.8, 1.0]
        baseline = [1.0, 1.0, 1.0, 1.0]
        result = paired_metric_statistics(
            proposed,
            baseline,
            higher_is_better=True,
            clusters=["seed-a", "seed-a", "seed-b", "seed-b"],
            samples=200,
            seed=4,
        )
        self.assertEqual(result["pairs"], 4)
        self.assertEqual(result["independent_clusters"], 2)
        self.assertAlmostEqual(result["raw_mean_difference"], 0.5)
        self.assertAlmostEqual(
            result["cluster_mean_favorable_difference"], 0.5
        )

    def test_nearly_constant_differences_do_not_emit_precision_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = paired_metric_statistics(
                [2.0, 2.0 + 1e-14],
                [1.0, 1.0],
                higher_is_better=True,
                samples=50,
                seed=4,
            )
        self.assertTrue(np.isnan(result["paired_t_test_p_value"]))
        self.assertTrue(np.isnan(result["cohen_dz_favorable"]))

    def test_scenario_slice_rows_cover_every_actor(self):
        scenario = generate_synthetic_scenario(8, slots=20, vehicles=4)
        rows = classify_scenario_slices(scenario)
        self.assertEqual(len(rows), scenario.vehicle_count)
        self.assertTrue(all(row.labels for row in rows))
        self.assertTrue(all(np.isfinite(row.range_change_m) for row in rows))


if __name__ == "__main__":
    unittest.main()
