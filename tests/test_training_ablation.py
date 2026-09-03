import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.learning.ablation import (
    CANONICAL_SEEDS,
    build_training_ablation_plan,
)


class TrainingAblationTest(unittest.TestCase):
    def test_plan_requires_five_frozen_seeds_and_hashes_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "samples.npz"
            dataset.write_bytes(b"immutable-sample-fixture")
            plan = build_training_ablation_plan(dataset, CANONICAL_SEEDS, epochs=80)
        self.assertEqual(plan.planned_runs, 20)
        self.assertEqual(len(plan.dataset_sha256), 64)
        self.assertEqual(plan.seeds, CANONICAL_SEEDS)

    def test_plan_rejects_three_seed_legacy_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "samples.npz"
            dataset.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                build_training_ablation_plan(dataset, (1, 2, 3), epochs=80)

    def test_plan_rejects_wrong_five_seed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "samples.npz"
            dataset.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                build_training_ablation_plan(dataset, (1, 2, 3, 4, 5), epochs=80)


if __name__ == "__main__":
    unittest.main()
