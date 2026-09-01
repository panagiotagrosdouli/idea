import unittest

from predictive_pc_fmcw.learning.lambda_sweep import default_lambda_settings


class LambdaSweepTest(unittest.TestCase):
    def test_default_sweep_is_one_axis_and_deduplicated(self):
        settings = default_lambda_settings()
        self.assertEqual(len(settings), 5)
        self.assertEqual(len(set(settings)), 5)
        self.assertIn((0.2, 0.1), {
            (item.lambda_link, item.lambda_outage) for item in settings
        })
        self.assertTrue(all(
            item.lambda_link == 0.2 or item.lambda_outage == 0.1
            for item in settings
        ))


if __name__ == "__main__":
    unittest.main()
