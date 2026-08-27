import unittest

from predictive_pc_fmcw.ber import simulate_dbpsk_ber


class BERTest(unittest.TestCase):
    def test_monte_carlo_tracks_dbpsk_theory(self):
        points = simulate_dbpsk_ber([0.0, 4.0], bits=80_000, seed=11)
        for point in points:
            self.assertLess(abs(point.simulated_ber - point.theoretical_ber), 0.015)
        self.assertLess(points[1].simulated_ber, points[0].simulated_ber)


if __name__ == "__main__":
    unittest.main()

