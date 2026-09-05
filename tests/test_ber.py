import unittest

from predictive_pc_fmcw.ber import (
    simulate_dbpsk_ber,
    simulate_part_a_notebook_receiver_ber,
)


class BERTest(unittest.TestCase):
    def test_monte_carlo_tracks_dbpsk_theory(self):
        points = simulate_dbpsk_ber([0.0, 4.0], bits=80_000, seed=11)
        for point in points:
            self.assertLess(abs(point.simulated_ber - point.theoretical_ber), 0.015)
        self.assertLess(points[1].simulated_ber, points[0].simulated_ber)

    def test_part_a_notebook_receiver_improves_with_waveform_snr(self):
        points = simulate_part_a_notebook_receiver_ber(
            [-5.0, 12.0], bits=4_000, seed=19
        )
        self.assertEqual(
            {point.receiver for point in points},
            {"supplied_part_a_fft_dpsk"},
        )
        self.assertGreater(points[0].simulated_ber, points[1].simulated_ber)
        self.assertLess(points[1].simulated_ber, 0.5)
        self.assertEqual(points[1].snr_semantics, "waveform_sample_snr_db")
        self.assertGreaterEqual(points[0].ber_for_lut, points[1].ber_for_lut)

    def test_part_a_receiver_does_not_suffer_alias_branch_burst(self):
        # These independent chirp seeds previously produced exactly 214/1000
        # errors because projection and compensation used different FFT alias
        # branches across half-sample symbol-centre transitions.
        for seed in (161963140, 3885075678, 3519752463):
            point = simulate_part_a_notebook_receiver_ber(
                [5.0], bits=1_000, seed=seed
            )[0]
            self.assertLess(point.simulated_ber, 0.05)


if __name__ == "__main__":
    unittest.main()
