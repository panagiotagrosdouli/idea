import csv
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.ber import BERPoint, write_ber_lut
from predictive_pc_fmcw.link_verification import verify_lut


class PartALinkVerificationTest(unittest.TestCase):
    def _canonical_points(self, bits: int = 250_000) -> list[BERPoint]:
        points = [
            BERPoint(
                ebn0_db=float(snr),
                simulated_ber=max(0.0, 0.3 - 0.01 * index),
                theoretical_ber=0.0,
                bits=bits,
                errors=round(max(0.0, 0.3 - 0.01 * index) * bits),
                ber_upper_95=0.001,
                ber_for_lut=max(0.001, 0.3 - 0.01 * index),
                receiver="supplied_part_a_fft_dpsk",
                snr_semantics="waveform_sample_snr_db",
            )
            for index, snr in enumerate(range(-5, 26))
        ]
        return points

    def test_canonical_shape_and_monotonicity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            write_ber_lut(self._canonical_points(), path)
            self.assertEqual(verify_lut(path)["status"], "PASS")

    def test_low_bit_count_fails_canonical_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            write_ber_lut(self._canonical_points(bits=100_000), path)
            report = verify_lut(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["minimum_bits_per_point_met"])

    def test_missing_columns_fails_without_crashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            path.write_text("ebn0_db,bits\n-5,250000\n", encoding="utf-8")
            report = verify_lut(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("receiver", report["missing_columns"])

    def test_nonmonotone_lut_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            fieldnames = [
                "ebn0_db",
                "simulated_ber",
                "theoretical_ber",
                "bits",
                "errors",
                "ber_upper_95",
                "ber_for_lut",
                "receiver",
                "snr_semantics",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index, snr in enumerate(range(-5, 26)):
                    value = 0.2 - index * 0.005
                    if snr == 10:
                        value = 0.3
                    writer.writerow(
                        {
                            "ebn0_db": snr,
                            "simulated_ber": value,
                            "theoretical_ber": 0.0,
                            "bits": 250000,
                            "errors": 1,
                            "ber_upper_95": 0.001,
                            "ber_for_lut": value,
                            "receiver": "supplied_part_a_fft_dpsk",
                            "snr_semantics": "waveform_sample_snr_db",
                        }
                    )
            self.assertEqual(verify_lut(path)["status"], "FAIL")

    def test_significant_raw_ber_increase_fails_even_with_monotone_lut(self):
        points = self._canonical_points()
        points[12] = BERPoint(
            ebn0_db=7.0,
            simulated_ber=0.008,
            theoretical_ber=0.0,
            bits=250_000,
            errors=2_000,
            ber_upper_95=0.009,
            ber_for_lut=0.06,
            receiver="supplied_part_a_fft_dpsk",
            snr_semantics="waveform_sample_snr_db",
        )
        points[13] = BERPoint(
            ebn0_db=8.0,
            simulated_ber=0.05,
            theoretical_ber=0.0,
            bits=250_000,
            errors=12_500,
            ber_upper_95=0.052,
            ber_for_lut=0.05,
            receiver="supplied_part_a_fft_dpsk",
            snr_semantics="waveform_sample_snr_db",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            write_ber_lut(points, path)
            report = verify_lut(path)
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(
                report["checks"]["raw_ber_no_statistically_significant_increase"]
            )
            self.assertEqual(
                report["raw_ber_significant_increases"][0]["higher_snr_db"], 8.0
            )

    def test_small_raw_monte_carlo_reversal_does_not_fail(self):
        points = self._canonical_points()
        points[29] = BERPoint(
            ebn0_db=24.0,
            simulated_ber=0.00100,
            theoretical_ber=0.0,
            bits=250_000,
            errors=250,
            ber_upper_95=0.0012,
            ber_for_lut=0.0011,
            receiver="supplied_part_a_fft_dpsk",
            snr_semantics="waveform_sample_snr_db",
        )
        points[30] = BERPoint(
            ebn0_db=25.0,
            simulated_ber=0.00101,
            theoretical_ber=0.0,
            bits=250_000,
            errors=252,
            ber_upper_95=0.0012,
            ber_for_lut=0.0010,
            receiver="supplied_part_a_fft_dpsk",
            snr_semantics="waveform_sample_snr_db",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            write_ber_lut(points, path)
            report = verify_lut(path)
            self.assertTrue(
                report["checks"]["raw_ber_no_statistically_significant_increase"]
            )


if __name__ == "__main__":
    unittest.main()
