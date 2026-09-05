import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from predictive_pc_fmcw.data.manifest import sha256_file
from predictive_pc_fmcw.learning.ablation import validate_training_resume
from predictive_pc_fmcw.learning.train import TrainingResult
from predictive_pc_fmcw.persistence import (
    persist_completed_file_atomic,
    validate_completed_file,
)


class ResumeHardeningTest(unittest.TestCase):
    def _result(self, dataset: Path, checkpoint: Path, **overrides):
        values = {
            "checkpoint": str(checkpoint),
            "best_epoch": 3,
            "validation_loss": 1.0,
            "validation_ade_m": 1.0,
            "validation_fde_m": 2.0,
            "validation_trajectory_loss": 1.0,
            "validation_link_loss": 0.2,
            "validation_outage_loss": 0.1,
            "train_samples": 10,
            "validation_samples": 2,
            "objective": "full",
            "seed": 20260827,
            "dataset_sha256": sha256_file(dataset),
        }
        values.update(overrides)
        return TrainingResult(**values)

    def test_valid_result_and_checkpoint_is_safe_to_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "training.npz"
            dataset.write_bytes(b"dataset")
            run_dir = root / "full" / "seed_20260827"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best_comm_aware_gru.pt"
            checkpoint.write_bytes(b"checkpoint")
            result_path = run_dir / "training_result.json"
            result_path.write_text(
                json.dumps(asdict(self._result(dataset, checkpoint))), encoding="utf-8"
            )
            validation = validate_training_resume(
                result_path,
                expected_objective="full",
                expected_seed=20260827,
                expected_dataset_sha256=sha256_file(dataset),
                expected_run_dir=run_dir,
            )
            self.assertTrue(validation.valid)
            self.assertEqual(validation.reason, "verified_complete")

    def test_missing_checkpoint_requires_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "training.npz"
            dataset.write_bytes(b"dataset")
            run_dir = root / "full" / "seed_20260827"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best_comm_aware_gru.pt"
            result_path = run_dir / "training_result.json"
            result_path.write_text(
                json.dumps(asdict(self._result(dataset, checkpoint))), encoding="utf-8"
            )
            validation = validate_training_resume(
                result_path,
                expected_objective="full",
                expected_seed=20260827,
                expected_dataset_sha256=sha256_file(dataset),
                expected_run_dir=run_dir,
            )
            self.assertFalse(validation.valid)
            self.assertEqual(validation.reason, "checkpoint_missing")

    def test_corrupt_result_json_requires_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "training.npz"
            dataset.write_bytes(b"dataset")
            run_dir = root / "full" / "seed_20260827"
            run_dir.mkdir(parents=True)
            result_path = run_dir / "training_result.json"
            result_path.write_text("{bad json", encoding="utf-8")
            validation = validate_training_resume(
                result_path,
                expected_objective="full",
                expected_seed=20260827,
                expected_dataset_sha256=sha256_file(dataset),
                expected_run_dir=run_dir,
            )
            self.assertFalse(validation.valid)
            self.assertTrue(validation.reason.startswith("invalid_result_json:"))

    def test_metadata_mismatch_requires_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "training.npz"
            dataset.write_bytes(b"dataset")
            run_dir = root / "full" / "seed_20260827"
            run_dir.mkdir(parents=True)
            checkpoint = run_dir / "best_comm_aware_gru.pt"
            checkpoint.write_bytes(b"checkpoint")
            result_path = run_dir / "training_result.json"
            result_path.write_text(
                json.dumps(
                    asdict(self._result(dataset, checkpoint, objective="trajectory_only"))
                ),
                encoding="utf-8",
            )
            validation = validate_training_resume(
                result_path,
                expected_objective="full",
                expected_seed=20260827,
                expected_dataset_sha256=sha256_file(dataset),
                expected_run_dir=run_dir,
            )
            self.assertFalse(validation.valid)
            self.assertEqual(validation.reason, "objective_mismatch")

    def test_stale_temporary_file_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            stale = Path(temporary) / "shard.tfrecord.persisting"
            stale.write_bytes(b"partial")
            validation = validate_completed_file(stale, expected_size=7)
            self.assertFalse(validation.valid)
            self.assertEqual(validation.reason, "temporary_file")

    def test_successful_completed_persistence_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local = root / "shard.completed"
            final = root / "drive" / "shard.tfrecord"
            local.write_bytes(b"complete-shard")
            validation = persist_completed_file_atomic(
                local, final, expected_size=len(b"complete-shard")
            )
            self.assertTrue(validation.valid)
            self.assertEqual(final.read_bytes(), b"complete-shard")
            self.assertFalse(final.with_name(final.name + ".persisting").exists())


if __name__ == "__main__":
    unittest.main()
