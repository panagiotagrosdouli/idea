import unittest

import numpy as np

from predictive_pc_fmcw.config import LinkConfig
from predictive_pc_fmcw.learning.losses import communication_aware_loss
from predictive_pc_fmcw.link import LinkModel


class CommunicationLossTest(unittest.TestCase):
    def test_exact_prediction_has_zero_geometric_and_link_loss(self):
        target = np.asarray([[[30.0, 0.0], [32.0, 1.0], [34.0, 2.0]]])
        loss = communication_aware_loss(target, target, LinkModel(LinkConfig()))
        self.assertAlmostEqual(loss.trajectory, 0.0)
        self.assertAlmostEqual(loss.link, 0.0)
        self.assertGreaterEqual(loss.outage, 0.0)

    def test_link_error_increases_total_loss(self):
        target = np.asarray([[[30.0, 0.0], [32.0, 0.0]]])
        shifted = target + np.asarray([40.0, 15.0])
        model = LinkModel(LinkConfig())
        exact = communication_aware_loss(target, target, model)
        wrong = communication_aware_loss(shifted, target, model)
        self.assertGreater(wrong.total, exact.total)


if __name__ == "__main__":
    unittest.main()

