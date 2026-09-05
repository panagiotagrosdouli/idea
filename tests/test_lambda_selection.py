import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from predictive_pc_fmcw.data.manifest import sha256_file
from predictive_pc_fmcw.learning.ablation import CANONICAL_SEEDS
from predictive_pc_fmcw.learning.lambda_selection import (
    freeze_lambda_selection,
    load_lambda_selection,
    validate_lambda_selection,
)
from predictive_pc_fmcw.learning.train import TrainingResult


class LambdaSelectionTest(unittest.TestCase):
    def _sweep_fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "training.npz"
        dataset.write_bytes(b"training")
        dataset_hash = sha256_file(dataset)
        sweep = root / "sweep"
        sweep.mkdir()
        settings = [
            {"lambda_link": 0.05, "lambda_outage": 0.1},
            {"lambda_link": 0.2, "lambda_outage": 0.1},
        ]
        (sweep / "lambda_sweep_plan.json").write_text(
            json.dumps(
                {
                    "dataset_sha256": dataset_hash,
                    "selection_scope": "development_only",
                    "seeds": list(CANONICAL_SEEDS),
                    "settings": settings,
                }
            ),
            encoding="utf-8",
        )
        for setting in ("link_0p05_outage_0p1", "link_0p2_outage_0p1"):
            for seed in CANONICAL_SEEDS:
                run_dir = sweep / setting / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = run_dir / "best_comm_aware_gru.pt"
                checkpoint.write_bytes(b"checkpoint")
                result = TrainingResult(
                    checkpoint=str(checkpoint),
                    best_epoch=3,
                    validation_loss=1.0,
                    validation_ade_m=1.0,
                    validation_fde_m=2.0,
                    validation_trajectory_loss=1.0,
                    validation_link_loss=0.2,
                    validation_outage_loss=0.1,
                    train_samples=10,
                    validation_samples=2,
                    objective="full",
                    seed=seed,
                    dataset_sha256=dataset_hash,
                )
                (run_dir / "training_result.json").write_text(
                    json.dumps(asdict(result)), encoding="utf-8"
                )
        return dataset, sweep

    def test_complete_sweep_selection_passes_and_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, sweep = self._sweep_fixture(Path(temporary))
            report = freeze_lambda_selection(
                sweep,
                dataset,
                lambda_link=0.2,
                lambda_outage=0.1,
                rationale="Selected on development-only trade-off review.",
            )
            self.assertEqual(report["status"], "PASS")
            selection = Path(temporary) / "selection.json"
            selection.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(validate_lambda_selection(selection, dataset)["valid"])
            self.assertEqual(load_lambda_selection(selection, dataset), (0.2, 0.1))

    def test_unswept_pair_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, sweep = self._sweep_fixture(Path(temporary))
            report = freeze_lambda_selection(
                sweep,
                dataset,
                lambda_link=0.5,
                lambda_outage=0.2,
                rationale="Development-only choice.",
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["selected_pair_was_swept"])

    def test_incomplete_sweep_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, sweep = self._sweep_fixture(Path(temporary))
            next(sweep.glob("*/seed_*/best_comm_aware_gru.pt")).unlink()
            report = freeze_lambda_selection(
                sweep,
                dataset,
                lambda_link=0.2,
                lambda_outage=0.1,
                rationale="Development-only choice.",
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["sweep_complete"])

    def test_cached_selection_with_wrong_scope_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset, sweep = self._sweep_fixture(Path(temporary))
            report = freeze_lambda_selection(
                sweep,
                dataset,
                lambda_link=0.2,
                lambda_outage=0.1,
                rationale="Development-only choice.",
            )
            report["selection_scope"] = "official_validation"
            selection = Path(temporary) / "selection.json"
            selection.write_text(json.dumps(report), encoding="utf-8")
            validation = validate_lambda_selection(selection, dataset)
            self.assertFalse(validation["valid"])
            self.assertEqual(validation["reason"], "selection_not_development_only")


if __name__ == "__main__":
    unittest.main()
