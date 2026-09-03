import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.data.scenario import MotionScenario
from predictive_pc_fmcw.data.training_export import build_training_npz_from_scenarios


class TrainingExportContractTest(unittest.TestCase):
    def _scenario(self, scenario_id: str, actor_ids: tuple[str, ...]) -> MotionScenario:
        timestamps = np.arange(91, dtype=np.float64) * 0.1
        ego = np.column_stack((timestamps, np.zeros_like(timestamps)))
        vehicles = np.stack(
            [
                np.column_stack(
                    (timestamps + 5.0 + index, np.full_like(timestamps, index + 1.0))
                )
                for index, _ in enumerate(actor_ids)
            ],
            axis=1,
        )
        return MotionScenario(
            scenario_id=scenario_id,
            timestamps_s=timestamps,
            ego_positions_xy=ego,
            vehicle_positions_xy=vehicles,
            actor_ids=actor_ids,
            start_index=11,
            source="official_WOMD_test_fixture",
        )

    def test_internal_split_is_scenario_safe_for_multiple_actors(self):
        scenarios = [
            self._scenario("scenario_a", ("a1", "a2")),
            self._scenario("scenario_b", ("b1", "b2")),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.npz"
            build_training_npz_from_scenarios(
                scenarios,
                path,
                source="real_WOMD_v1.3.1_true_SDC_geometry",
            )
            with np.load(path, allow_pickle=False) as archive:
                scenario_ids = archive["scenario_id"].astype(str)
                splits = archive["split"].astype(str)
                for scenario_id in set(scenario_ids):
                    labels = set(splits[scenario_ids == scenario_id])
                    self.assertEqual(len(labels), 1)
                self.assertEqual(
                    str(archive["coordinate_frame"]),
                    "world_xy_with_explicit_ego_heading",
                )

    def test_official_validation_fixed_split_applies_to_every_actor(self):
        scenarios = [self._scenario("scenario_validation", ("v1", "v2"))]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.npz"
            build_training_npz_from_scenarios(
                scenarios,
                path,
                source="real_WOMD_v1.3.1_true_SDC_geometry",
                fixed_split="official_validation",
            )
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(set(archive["split"].astype(str)), {"official_validation"})
                self.assertEqual(
                    str(archive["source"]),
                    "real_WOMD_v1.3.1_true_SDC_geometry",
                )


if __name__ == "__main__":
    unittest.main()
