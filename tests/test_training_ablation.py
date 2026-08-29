import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.learning.ablation import build_training_ablation_plan


class TrainingAblationTest(unittest.TestCase):
    def test_plan_requires_multiple_seeds_and_hashes_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "samples.npz"
            dataset.write_bytes(b"immutable-sample-fixture")
            plan = build_training_ablation_plan(dataset, (1, 2, 3), epochs=5)
        self.assertEqual(plan.planned_runs, 12)
        self.assertEqual(len(plan.dataset_sha256), 64)

    def test_plan_rejects_single_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "samples.npz"
            dataset.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                build_training_ablation_plan(dataset, (1,), epochs=5)


if __name__ == "__main__":
    unittest.main()
