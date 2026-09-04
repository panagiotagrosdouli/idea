import unittest
from types import SimpleNamespace

from predictive_pc_fmcw.ber_diagnostics import run_chirp_cluster_diagnostic


class BERDiagnosticsTest(unittest.TestCase):
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
