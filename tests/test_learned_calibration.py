import unittest

import numpy as np

from predictive_pc_fmcw.learning.calibration import (
    fit_residual_gaussian,
    gaussian_nll_and_coverage,
)


class LearnedCalibrationTest(unittest.TestCase):
    def test_fits_positive_per_horizon_variance_and_reports_coverage(self):
        rng = np.random.default_rng(7)
        prediction = np.zeros((2000, 3, 2))
        target = rng.normal(size=prediction.shape) * np.asarray([1.0, 2.0])
        calibration = fit_residual_gaussian(prediction, target)
        self.assertEqual(calibration.variance_xy.shape, (3, 2))
        self.assertTrue(np.all(calibration.variance_xy > 0))
        metrics = gaussian_nll_and_coverage(prediction, target, calibration)
        self.assertEqual(metrics["nll"].shape, (2000, 3))
        self.assertGreater(float(metrics["coverage_95"].mean()), 0.92)
        self.assertLess(float(metrics["coverage_95"].mean()), 0.98)

    def test_rejects_invalid_variance(self):
        prediction = np.zeros((2, 3, 2))
        target = np.zeros_like(prediction)
        calibration = fit_residual_gaussian(prediction, target)
        object.__setattr__(calibration, "variance_xy", np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            gaussian_nll_and_coverage(prediction, target, calibration)


if __name__ == "__main__":
    unittest.main()
