from __future__ import annotations

import unittest
from types import SimpleNamespace

from predictive_pc_fmcw.ber import BERPoint
from predictive_pc_fmcw.ber_diagnostics import (
    paired_chirp_reversal_diagnostic,
    run_chirp_cluster_diagnostic,
)


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


class ChirpClusterDiagnosticTest(unittest.TestCase):
    @staticmethod
    def _receiver(snr, *, bits, seed):
        rate = 0.2 if seed % 5 == 0 else 0.001 * (10.0 - float(snr[0]))
        errors = round(max(0.0, rate) * bits)
        return [
            SimpleNamespace(
                simulated_ber=errors / bits,
                bits=bits,
                errors=errors,
            )
        ]

    def test_reports_independent_chirp_dispersion_and_is_deterministic(self):
        kwargs = {
            "trials_per_snr": 10,
            "decisions_per_trial": 1_000,
            "bootstrap_resamples": 200,
            "seed": 17,
            "receiver": self._receiver,
        }
        first = run_chirp_cluster_diagnostic([5.0, 8.0], **kwargs)
        second = run_chirp_cluster_diagnostic([5.0, 8.0], **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first["rows"]), 2)
        self.assertEqual(first["rows"][0]["independent_chirp_trials"], 10)
        self.assertEqual(len(first["rows"][0]["trials"]), 10)
        self.assertGreater(first["rows"][0]["max_chirp_ber"], 0.05)
        self.assertGreaterEqual(
            first["rows"][0]["cluster_bootstrap_ci_95"][1],
            first["rows"][0]["cluster_bootstrap_ci_95"][0],
        )

    def test_rejects_multi_chirp_decision_count(self):
        with self.assertRaises(ValueError):
            run_chirp_cluster_diagnostic(
                [5.0],
                trials_per_snr=2,
                decisions_per_trial=10_000,
                receiver=self._receiver,
            )


if __name__ == "__main__":
    unittest.main()
