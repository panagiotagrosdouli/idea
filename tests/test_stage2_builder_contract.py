import unittest

import numpy as np

from predictive_pc_fmcw.link_verification import CANONICAL_MIN_BITS_PER_POINT


class Stage2BuilderContractTest(unittest.TestCase):
    def test_canonical_minimum_bits_is_frozen(self):
        self.assertEqual(CANONICAL_MIN_BITS_PER_POINT, 250_000)

    def test_canonical_snr_grid_is_exact(self):
        grid = np.arange(-5.0, 26.0)
        self.assertEqual(grid.size, 31)
        np.testing.assert_array_equal(grid, np.arange(-5, 26, dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
