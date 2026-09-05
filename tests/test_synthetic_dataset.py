import json
from pathlib import Path

import numpy as np
import pytest

from predictive_pc_fmcw.synthetic.dataset import (
    DatasetBuildConfig,
    build_dataset,
    validate_dataset,
)


def test_small_dataset_materializes_and_validates(tmp_path: Path) -> None:
    config = DatasetBuildConfig(scenarios_per_family=1, ood_scenarios_per_family=1)
    manifest = build_dataset(tmp_path, config=config)
    report = validate_dataset(tmp_path)
    assert manifest["scenario_count"] == 22
    assert report["status"] == "PASS"
    assert report["scenario_count"] == 22
    assert len(report["split_sha256"]) == 64


def test_materialization_is_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = DatasetBuildConfig(scenarios_per_family=1, ood_scenarios_per_family=1)
    one = build_dataset(first, config=config)
    two = build_dataset(second, config=config)
    assert one["scenario_sha256"] == two["scenario_sha256"]
    assert one["split"]["sha256"] == two["split"]["sha256"]


def test_validator_rejects_modified_scenario(tmp_path: Path) -> None:
    config = DatasetBuildConfig(scenarios_per_family=1, ood_scenarios_per_family=1)
    manifest = build_dataset(tmp_path, config=config)
    scenario_id = next(iter(manifest["scenario_sha256"]))
    path = tmp_path / "scenarios" / f"{scenario_id}.npz"
    with path.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_dataset(tmp_path)


def test_link_ground_truth_is_present_and_bounded(tmp_path: Path) -> None:
    config = DatasetBuildConfig(scenarios_per_family=1, ood_scenarios_per_family=1)
    manifest = build_dataset(tmp_path, config=config)
    scenario_id = manifest["split"]["train"][0]
    path = tmp_path / "scenarios" / f"{scenario_id}.npz"
    with np.load(path, allow_pickle=True) as data:
        assert np.all((data["ber"] >= 0.0) & (data["ber"] <= 0.5))
        assert np.all((data["per"] >= 0.0) & (data["per"] <= 1.0))
        assert np.isfinite(float(data["link_lifetime_s"]))


def test_manifest_declares_no_external_dataset(tmp_path: Path) -> None:
    build_dataset(
        tmp_path,
        config=DatasetBuildConfig(scenarios_per_family=1, ood_scenarios_per_family=1),
    )
    raw = json.loads((tmp_path / "manifest.json").read_text())
    assert raw["scientific_guards"]["external_trajectory_dataset"] is False
    assert raw["scientific_guards"]["held_out_for_selection"] is False
