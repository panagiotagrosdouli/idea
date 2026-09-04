from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "womd_paper_corpus.json"


def test_canonical_womd_manifest_is_frozen() -> None:
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert spec["dataset"] == "Waymo Open Motion Dataset"
    assert spec["version"] == "1.3.1"
    assert spec["geometry"] == "true_SDC"
    assert spec["history_steps"] == 11
    assert spec["future_steps"] == 80
    assert spec["max_vehicles_per_scenario"] == 16
    assert spec["training"]["total_shards"] == 1000
    assert spec["training"]["selected_shard_start"] == 0
    assert spec["training"]["selected_shard_stop_exclusive"] == 50
    assert spec["validation"]["total_shards"] == 150
    assert spec["validation"]["selected_shard_start"] == 0
    assert spec["validation"]["selected_shard_stop_exclusive"] == 40
    assert spec["validation"]["role"] == "official_validation"
    assert spec["policy"]["scenario_overlap_allowed"] is False
    assert spec["policy"]["validation_for_model_selection"] is False
    assert spec["policy"]["force_historical_counts"] is False


def test_historical_fingerprint_remains_provenance_reference() -> None:
    spec = json.loads(CONFIG.read_text(encoding="utf-8"))
    fingerprint = spec["historical_training_fingerprint"]
    assert fingerprint == {
        "sample_count": 249137,
        "unique_scenarios": 24182,
        "sha256": "b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee",
    }
