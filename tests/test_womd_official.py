import unittest
from types import SimpleNamespace

from predictive_pc_fmcw.data.womd_official import (
    scenario_proto_to_motion_scenario,
)


def _track(track_id, offset, object_type=1, invalid_index=None):
    states = [
        SimpleNamespace(
            center_x=float(step + offset),
            center_y=float(offset),
            valid=step != invalid_index,
        )
        for step in range(6)
    ]
    return SimpleNamespace(id=track_id, object_type=object_type, states=states)


class OfficialWOMDTest(unittest.TestCase):
    def test_true_sdc_and_valid_vehicle_filter(self):
        proto = SimpleNamespace(
            scenario_id="official-1",
            timestamps_seconds=[0.1 * step for step in range(6)],
            current_time_index=2,
            sdc_track_index=1,
            tracks=[
                _track("pedestrian", 2.0, object_type=2),
                _track("sdc", 0.0),
                _track("valid-car", 5.0),
                _track("invalid-car", 3.0, invalid_index=4),
            ],
        )
        scenario = scenario_proto_to_motion_scenario(proto)
        self.assertIsNotNone(scenario)
        self.assertEqual(scenario.actor_ids, ("valid-car",))
        self.assertEqual(scenario.start_index, 3)
        self.assertEqual(scenario.source, "real_WOMD_v1.3.0_true_SDC_model_based_link")
        self.assertAlmostEqual(scenario.ego_positions_xy[2, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
