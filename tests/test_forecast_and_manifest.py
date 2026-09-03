import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.config import LinkConfig
from predictive_pc_fmcw.data.manifest import (
    audit_scenario_overlap,
    build_compact_womd_manifest,
    deterministic_development_split,
)
from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.forecast_evaluation import (
    _binary_auroc,
    evaluate_motion_and_link_forecasts,
    summarize_forecasts,
)
from predictive_pc_fmcw.link import LinkModel


class ForecastAndManifestTest(unittest.TestCase):
    def test_auroc_uses_all_positive_negative_pairs(self):
        labels = [False, True, False, True]
        scores = [0.1, 0.8, 0.2, 0.9]
        self.assertAlmostEqual(_binary_auroc(labels, scores), 1.0)

    def test_oracle_has_zero_motion_and_link_error(self):
        scenario = generate_synthetic_scenario(31, slots=8, vehicles=2)
        rows = evaluate_motion_and_link_forecasts(
            [scenario],
            LinkModel(LinkConfig()),
            horizon_steps=3,
            predictors={"oracle": None},
            anchor_stride=2,
        )
        summary = summarize_forecasts(rows)["oracle"]
        self.assertAlmostEqual(summary["ade_m"], 0.0, places=10)
        self.assertAlmostEqual(summary["fde_m"], 0.0, places=10)
        self.assertAlmostEqual(summary["range_mae_m"], 0.0, places=10)
        self.assertAlmostEqual(summary["snr_mae_db"], 0.0, places=10)
        self.assertAlmostEqual(summary["link_lifetime_error_steps"], 0.0, places=10)
        self.assertAlmostEqual(summary["outage_f1"], 1.0, places=10)

    def test_manifest_hash_and_scenario_split_are_deterministic(self):
        records = [
            {
                "scenario_id": scenario_id,
                "track_index": actor,
                "past": [[0.0, 0.0], [1.0, 0.0]],
                "future": [[2.0, 0.0]],
            }
            for scenario_id in ("scene-a", "scene-b")
            for actor in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "compact.json"
            source.write_text(json.dumps(records), encoding="utf-8")
            first = build_compact_womd_manifest(source)
            second = build_compact_womd_manifest(source)
        self.assertEqual(first, second)
        self.assertEqual(first["record_count"], 4)
        self.assertEqual(first["scenario_count"], 2)
        self.assertEqual(
            deterministic_development_split("scene-a"),
            deterministic_development_split("scene-a"),
        )

    def test_npz_overlap_audit_passes_and_rejects_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.npz"
            heldout = root / "heldout.npz"
            np.savez(train, scenario_id=np.asarray(["a", "a", "b"]))
            np.savez(heldout, scenario_id=np.asarray(["c"]))
            clean = audit_scenario_overlap({"train": train, "heldout": heldout})
            self.assertTrue(clean["passed"])
            np.savez(heldout, scenario_id=np.asarray(["b", "c"]))
            leaked = audit_scenario_overlap({"train": train, "heldout": heldout})
            self.assertFalse(leaked["passed"])
            self.assertEqual(
                leaked["pairwise_overlaps"]["heldout__train"]["scenario_ids"],
                ["b"],
            )


if __name__ == "__main__":
    unittest.main()
