import unittest

import numpy as np

from predictive_pc_fmcw.geometry import (
    heading_from_positions,
    range_and_bearing,
    wrap_angle_rad,
)


class GeometryTest(unittest.TestCase):
    def test_range_and_ego_relative_bearing(self):
        distance, bearing = range_and_bearing([3, 4], [0, 0], np.pi / 2)
        self.assertAlmostEqual(float(distance), 5.0)
        self.assertAlmostEqual(float(bearing), np.arctan2(4, 3) - np.pi / 2)

    def test_heading_and_angle_wrap(self):
        heading = heading_from_positions([[0, 0], [1, 0], [1, 1]])
        np.testing.assert_allclose(heading, [0.0, 0.0, np.pi / 2])
        self.assertAlmostEqual(float(wrap_angle_rad(3 * np.pi)), -np.pi)


if __name__ == "__main__":
    unittest.main()

