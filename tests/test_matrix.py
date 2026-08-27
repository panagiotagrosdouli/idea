import tempfile
import unittest

from predictive_pc_fmcw.config import ExperimentConfig
from predictive_pc_fmcw.experiment_matrix import run_experiment_matrix, write_matrix


class MatrixTest(unittest.TestCase):
    def test_minimal_matrix(self):
        matrix = {
            "prediction_horizon_steps": [3],
            "vehicles": [2],
            "offered_load": [0.5],
            "slot_duration_s": [0.1],
            "seeds": [10],
        }
        rows = run_experiment_matrix(
            ExperimentConfig(), matrix, schedulers=("reactive_greedy", "oracle")
        )
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_matrix(rows, directory)
            self.assertTrue(all(path.exists() for path in artifacts.values()))


if __name__ == "__main__":
    unittest.main()

