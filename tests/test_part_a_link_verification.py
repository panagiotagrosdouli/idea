import csv
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.ber import BERPoint, write_ber_lut
from predictive_pc_fmcw.link_verification import verify_lut


class PartALinkVerificationTest(unittest.TestCase):
    def test_canonical_shape_and_monotonicity(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lut.csv"
            points = [
                BERPoint(
                    ebn0_db=float(snr),
                    simulated_ber=max(0.0, 0.3 - 0.01 * index),
                    theoretical_ber=0.0,
                    bits=250_000,
                    errors=1,
                    ber_upper_95=0.001,
                    ber_for_lut=max(0.001, 0.3 - 0.01 * index),
                    receiver="supplied_part_a_fft_dpsk",
                    snr_semantics="waveform_sample_snr_db",
                )
                for index, snr in enumerate(range(-5, 26))
            ]
            write_ber_lut(points, path)
            self.assertEqual(verify_lut(path)["status"], "PASS")

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


if __name__ == "__main__":
    unittest.main()
