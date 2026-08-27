import unittest

import numpy as np

from predictive_pc_fmcw.config import LinkConfig
from predictive_pc_fmcw.link import LinkModel


class LinkModelTest(unittest.TestCase):
    def setUp(self):
        self.model = LinkModel(LinkConfig())

    def test_distance_and_pointing_reduce_snr(self):
        distance = self.model.evaluate_arrays([20, 40, 80], [0, 0, 0])["snr_db"]
        pointing = self.model.evaluate_arrays(
            [40, 40, 40], np.deg2rad([0, 10, 20])
        )["snr_db"]
        self.assertTrue(np.all(np.diff(distance) < 0))
        self.assertTrue(np.all(np.diff(pointing) < 0))

    def test_ber_and_per_are_bounded_and_monotone(self):
        ber = self.model.dbpsk_ber([0.1, 1, 10])
        per = self.model.packet_error_rate(ber)
        self.assertTrue(np.all(np.diff(ber) < 0))
        self.assertTrue(np.all((per >= 0) & (per <= 1)))
        self.assertTrue(np.all(per >= ber))

    def test_reference_snr(self):
        state = self.model.evaluate(40.0, 0.0)
        self.assertAlmostEqual(state.snr_db, 18.0, places=8)

    def test_outside_fov_is_outage(self):
        state = self.model.evaluate(20.0, np.deg2rad(80))
        self.assertTrue(state.outage)


if __name__ == "__main__":
    unittest.main()

