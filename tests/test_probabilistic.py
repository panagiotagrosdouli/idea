import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
)
from predictive_pc_fmcw.probabilistic import (
    evaluate_probabilistic_baselines,
    write_probabilistic_artifacts,
)


class ProbabilisticCalibrationTest(unittest.TestCase):
    def test_calibration_is_scenario_safe_and_bounded(self):
        calibration = [
            generate_synthetic_scenario(seed, slots=24, vehicles=2)
            for seed in (11, 12)
        ]
        evaluation = [
            generate_synthetic_scenario(seed, slots=24, vehicles=2)
            for seed in (21, 22)
        ]
        fitted, rows = evaluate_probabilistic_baselines(
            calibration,
            evaluation,
            {
                "gaussian_cv": ConstantVelocityPredictor(),
                "gaussian_ca": ConstantAccelerationPredictor(),
            },
            horizon_steps=5,
            anchor_stride=3,
        )
        self.assertEqual(len(fitted), 2)
        self.assertEqual(len(rows), 2)
        for calibration_row, row in zip(fitted, rows, strict=True):
            self.assertEqual(row.calibration_evaluation_overlap, 0)
            self.assertGreater(row.samples, 0)
            self.assertGreater(row.mean_sigma_m, 0)
            self.assertTrue(
                all(
                    later >= earlier
                    for earlier, later in zip(
                        calibration_row.isotropic_variance_m2,
                        calibration_row.isotropic_variance_m2[1:],
                        strict=False,
                    )
                )
            )
            for value in (row.coverage_50, row.coverage_90, row.coverage_95):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_probabilistic_artifact_declares_no_learned_checkpoint(self):
        calibration = [generate_synthetic_scenario(31, slots=20, vehicles=2)]
        evaluation = [generate_synthetic_scenario(41, slots=20, vehicles=2)]
        fitted, rows = evaluate_probabilistic_baselines(
            calibration,
            evaluation,
            {"gaussian_cv": ConstantVelocityPredictor()},
            horizon_steps=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_probabilistic_artifacts(fitted, rows, directory)
            payload = Path(paths["json"]).read_text(encoding="utf-8")
            self.assertIn('"learned_gaussian_or_gmm_checkpoint_used": false', payload)


if __name__ == "__main__":
    unittest.main()
