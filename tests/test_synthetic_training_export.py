from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from predictive_pc_fmcw.synthetic.dataset import DatasetBuildConfig, build_dataset
from predictive_pc_fmcw.synthetic.training_export import (
    build_synthetic_training_npz,
    validate_synthetic_training_npz,
)


def _small_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    build_dataset(
        root,
        config=DatasetBuildConfig(
            scenarios_per_family=2,
            ood_scenarios_per_family=1,
        ),
    )
    return root


def test_export_contains_only_train_and_development(tmp_path: Path) -> None:
    root = _small_dataset(tmp_path)
    output = tmp_path / "training.npz"
    build_synthetic_training_npz(root, output, history_steps=8, horizon_steps=4)
    report = validate_synthetic_training_npz(root, output)
    assert report["status"] == "PASS"
    assert report["held_out_or_ood_samples"] == 0
    with np.load(output, allow_pickle=False) as data:
        assert set(np.unique(data["split"]).tolist()) == {"training", "development"}
        manifest = json.loads((root / "manifest.json").read_text())
        forbidden = set(manifest["split"]["held_out_test"]) | set(
            manifest["split"]["ood_test"]
        )
        assert set(data["scenario_id"].astype(str)).isdisjoint(forbidden)


def test_history_is_derived_from_noisy_causal_observations(tmp_path: Path) -> None:
    root = _small_dataset(tmp_path)
    output = tmp_path / "training.npz"
    build_synthetic_training_npz(root, output, history_steps=8, horizon_steps=4)
    with np.load(output, allow_pickle=False) as exported:
        scenario_id = str(exported["scenario_id"][0])
        end_index = int(exported["sample_history_end_index"][0])
        history = exported["history_xy"][0]
    with np.load(root / "scenarios" / f"{scenario_id}.npz", allow_pickle=False) as source:
        observed_range = source["observed_range_m"]
        observed_bearing = source["observed_bearing_rad"]
        observed_xy = np.stack(
            (
                observed_range * np.cos(observed_bearing),
                observed_range * np.sin(observed_bearing),
            ),
            axis=-1,
        )
    np.testing.assert_allclose(history, observed_xy[end_index - 7 : end_index + 1])


def test_validator_rejects_forbidden_scenario(tmp_path: Path) -> None:
    root = _small_dataset(tmp_path)
    output = tmp_path / "training.npz"
    build_synthetic_training_npz(root, output, history_steps=8, horizon_steps=4)
    manifest = json.loads((root / "manifest.json").read_text())
    forbidden_id = manifest["split"]["held_out_test"][0]
    with np.load(output, allow_pickle=False) as data:
        payload = {name: data[name] for name in data.files}
    payload["scenario_id"] = payload["scenario_id"].copy()
    payload["scenario_id"][0] = forbidden_id
    corrupted = tmp_path / "corrupted.npz"
    np.savez_compressed(corrupted, **payload)
    with pytest.raises(ValueError, match="held-out/OOD contamination"):
        validate_synthetic_training_npz(root, corrupted)
