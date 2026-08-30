import tempfile
import unittest

from predictive_pc_fmcw.complexity import (
    measure_complexity,
    write_complexity_artifacts,
)
from predictive_pc_fmcw.config import ExperimentConfig


class ComplexityTest(unittest.TestCase):
    def test_complexity_rows_cover_predictors_schedulers_and_gru_count(self):
        rows = measure_complexity(ExperimentConfig(), repeats=2, warmup=0)
        kinds = {row.kind for row in rows}
        self.assertIn("trajectory_predictor", kinds)
        self.assertIn("scheduler", kinds)
        learned = [
            row for row in rows if row.kind == "learned_predictor_unexecuted"
        ]
        self.assertEqual(len(learned), 1)
        self.assertGreater(learned[0].parameters, 0)
        with tempfile.TemporaryDirectory() as directory:
            artifacts = write_complexity_artifacts(rows, directory)
            self.assertTrue(all(path.exists() for path in artifacts.values()))


if __name__ == "__main__":
    unittest.main()
