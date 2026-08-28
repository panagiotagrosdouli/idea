import csv
import tempfile
import unittest
from pathlib import Path

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

    def test_channel_ablation_modes(self):
        range_only = LinkModel(LinkConfig(channel_mode="range_only"))
        range_pointing = LinkModel(LinkConfig(channel_mode="range_pointing"))
        full = LinkModel(LinkConfig(channel_mode="full"))
        on_axis = range_only.evaluate(80.0, 0.0)
        off_axis = range_only.evaluate(80.0, np.deg2rad(80.0))
        self.assertAlmostEqual(on_axis.snr_db, off_axis.snr_db)
        self.assertGreater(
            range_pointing.evaluate(80.0, 0.0).snr_db,
            full.evaluate(80.0, 0.0).snr_db,
        )

    def test_part_a_ber_lut_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lut.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "ebn0_db",
                        "simulated_ber",
                        "theoretical_ber",
                        "bits",
                        "errors",
                    ),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "ebn0_db": 0,
                            "simulated_ber": 0.2,
                            "theoretical_ber": 0.2,
                            "bits": 1000,
                            "errors": 200,
                        },
                        {
                            "ebn0_db": 20,
                            "simulated_ber": 1e-6,
                            "theoretical_ber": 1e-6,
                            "bits": 1000,
                            "errors": 0,
                        },
                    ]
                )
            model = LinkModel(
                LinkConfig(ber_source="lut", ber_lut_path=str(path))
            )
            values = model.evaluate_arrays([40.0, 80.0], [0.0, 0.0])
        self.assertTrue(np.all(np.isfinite(values["ber"])))
        self.assertTrue(np.all((values["ber"] > 0) & (values["ber"] <= 0.5)))


if __name__ == "__main__":
    unittest.main()
