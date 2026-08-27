import unittest

from predictive_pc_fmcw.config import ExperimentConfig
from predictive_pc_fmcw.validation import run_validation


class ValidationTest(unittest.TestCase):
    def test_all_scientific_gates_pass(self):
        report = run_validation(ExperimentConfig())
        self.assertEqual(report.status, "PASS")


if __name__ == "__main__":
    unittest.main()

