from __future__ import annotations

import unittest

from predictive_pc_fmcw.ber import BERPoint
from predictive_pc_fmcw.ber_diagnostics import paired_chirp_reversal_diagnostic


def point(snr: float, ber: float, bits: int) -> BERPoint:
    return BERPoint(snr, ber, 0.0, bits, round(ber * bits), ber, ber, "test", "test")


class PairedChirpDiagnosticTest(unittest.TestCase):
    def test_common_seed_is_used_for_both_snr_values(self):
        calls: list[tuple[float, int]] = []

        def estimator(grid, *, bits, seed):
            snr = float(grid[0])
            calls.append((snr, seed))
            return [point(snr, 0.1 if snr == 7 else 0.05, bits)]

        report = paired_chirp_reversal_diagnostic(
            7, 8, trials=3, bootstrap_repetitions=100, estimator=estimator
        )
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[2][1], calls[3][1])
        self.assertFalse(report["higher_snr_worse_supported"])

    def test_material_paired_reversal_is_detected(self):
        def estimator(grid, *, bits, seed):
            snr = float(grid[0])
            return [point(snr, 0.01 if snr == 7 else 0.04, bits)]

        report = paired_chirp_reversal_diagnostic(
            7, 8, trials=4, bootstrap_repetitions=100, estimator=estimator
        )
        self.assertTrue(report["higher_snr_worse_supported"])
        self.assertTrue(report["material_reversal_supported"])
        self.assertEqual(report["sampling_unit"], "independent_chirp")

    def test_invalid_design_is_rejected(self):
        with self.assertRaises(ValueError):
            paired_chirp_reversal_diagnostic(8, 7)


if __name__ == "__main__":
    unittest.main()
