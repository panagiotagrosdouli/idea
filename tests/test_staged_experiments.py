import unittest

from predictive_pc_fmcw.config import ExperimentConfig
from predictive_pc_fmcw.staged_experiments import _staged_settings


class StagedExperimentTest(unittest.TestCase):
    def test_slot_study_preserves_physical_horizon_and_duration(self):
        config = ExperimentConfig()
        expected_horizon = (
            config.prediction_horizon_steps * config.slot_duration_s
        )
        settings = [
            current
            for study, _, current in _staged_settings(config, seed=1)
            if study == "slot_duration"
        ]
        self.assertEqual(len(settings), 3)
        for current in settings:
            self.assertAlmostEqual(
                current.prediction_horizon_steps * current.slot_duration_s,
                expected_horizon,
            )
            self.assertEqual(
                current.benchmark.duration_s, config.benchmark.duration_s
            )
            self.assertEqual(current.traffic.deadline_s, config.traffic.deadline_s)


if __name__ == "__main__":
    unittest.main()
