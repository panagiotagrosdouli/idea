import json
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.data.manifest import sha256_file
from predictive_pc_fmcw.learning.ablation import CANONICAL_SEEDS
from predictive_pc_fmcw.learning.lambda_selection import (
    freeze_lambda_selection,
    load_lambda_selection,
)


class LambdaSelectionTest(unittest.TestCase):
    def _sweep_fixture(self, root: Path) -> tuple[Path, Path]:
        dataset = root / "training.npz"
        dataset.write_bytes(b"training")
        sweep = root / "sweep"
        sweep.mkdir()
        settings = [
            {"lambda_link": 0.05, "lambda_outage": 0.1},
            {"lambda_link": 0.2, "lambda_outage": 0.1},
        ]
        (sweep / "lambda_sweep_plan.json").write_text(
            json.dumps(
                {
                    "dataset_sha256": sha256_file(dataset),
                    "seeds": list(CANONICAL_SEEDS),
                    "settings": settings,
                }
            ),
            encoding="utf-8",
        )
        for setting in ("link_0p05_outage_0p1", "link_0p2_outage_0p1"):
            for seed in CANONICAL_SEEDS:
                path = sweep / setting / f"seed_{seed}" / "training_result.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
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
            next(sweep.glob("*/seed_*/training_result.json")).unlink()
            report = freeze_lambda_selection(
                sweep,
                dataset,
                lambda_link=0.2,
                lambda_outage=0.1,
                rationale="Development-only choice.",
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["checks"]["sweep_complete"])


if __name__ == "__main__":
    unittest.main()
