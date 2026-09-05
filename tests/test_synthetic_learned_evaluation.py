from pathlib import Path

import numpy as np
import pytest

from predictive_pc_fmcw.learning.ablation import CANONICAL_SEEDS, OBJECTIVES
from predictive_pc_fmcw.synthetic.learned_evaluation import (
    METRIC_NAMES,
    _objective_summary,
    _validate_official_npz,
)


def _freeze() -> dict[str, object]:
    return {
        "training_npz_sha256": "train-sha",
        "dataset_manifest_sha256": "dataset-sha",
        "completion_manifest_sha256": "completion-sha",
    }


def _write_official(path: Path, *, training_hash: str = "train-sha") -> None:
    np.savez_compressed(
        path,
        history_xy=np.zeros((2, 3, 2), dtype=np.float32),
        future_xy=np.zeros((2, 2, 2), dtype=np.float32),
        future_ego_heading_rad=np.zeros((2, 2), dtype=np.float32),
        scenario_id=np.asarray(["a", "b"]),
        split=np.asarray(["held_out_test", "held_out_test"]),
        freeze_training_npz_sha256=np.asarray(training_hash),
        freeze_dataset_manifest_sha256=np.asarray("dataset-sha"),
        freeze_completion_manifest_sha256=np.asarray("completion-sha"),
    )


def test_official_npz_requires_matching_freeze_hashes(tmp_path: Path) -> None:
    path = tmp_path / "official.npz"
    _write_official(path)
    split, arrays = _validate_official_npz(path, freeze=_freeze())
    assert split == "held_out_test"
    assert arrays["history_xy"].shape == (2, 3, 2)

    bad = tmp_path / "bad.npz"
    _write_official(bad, training_hash="wrong")
    with pytest.raises(PermissionError, match="training freeze hash mismatch"):
        _validate_official_npz(bad, freeze=_freeze())


def test_objective_summary_requires_all_five_seeds_per_scenario() -> None:
    rows: list[dict[str, object]] = []
    for objective_index, objective in enumerate(OBJECTIVES):
        for scenario_index, scenario_id in enumerate(("scenario-a", "scenario-b")):
            for seed_index, seed in enumerate(CANONICAL_SEEDS):
                base = float(objective_index + scenario_index + seed_index)
                row: dict[str, object] = {
                    "objective": objective,
                    "scenario_id": scenario_id,
                    "seed": seed,
                }
                for metric in METRIC_NAMES:
                    row[metric] = base
                rows.append(row)
    summary = _objective_summary(rows)
    assert set(summary) == set(OBJECTIVES)
    for objective in OBJECTIVES:
        assert summary[objective]["scenario_count"] == 2
        assert summary[objective]["seed_count"] == 5

    rows.pop()
    with pytest.raises(RuntimeError, match="all five seeds"):
        _objective_summary(rows)
