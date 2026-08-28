from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_development_split(
    scenario_id: str, development_fraction: float = 0.10
) -> str:
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be in (0, 1).")
    digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "development" if value < development_fraction else "training"


def build_compact_womd_manifest(
    womd_json: str | Path,
    release: str = "unknown-supplied-compact-export",
    development_fraction: float = 0.10,
) -> dict[str, object]:
    source = Path(womd_json)
    with source.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    scenario_ids = sorted({str(record["scenario_id"]) for record in records})
    split_by_scenario = {
        scenario_id: deterministic_development_split(
            scenario_id, development_fraction
        )
        for scenario_id in scenario_ids
    }
    return {
        "dataset": "Waymo Open Motion Dataset compact trajectory export",
        "release": release,
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "license": "Waymo Dataset License Agreement for Non-Commercial Use",
        "contains_official_sdc_index": False,
        "contains_raw_lidar": False,
        "record_count": len(records),
        "scenario_count": len(scenario_ids),
        "development_fraction": development_fraction,
        "split_method": "SHA256(scenario_id), deterministic",
        "split_by_scenario": split_by_scenario,
        "limitations": [
            "The originating official WOMD release is absent from the export.",
            "The compact records omit sdc_track_index and map context.",
            "A proxy ego is used only for software and mechanism evaluation.",
        ],
    }


def write_compact_womd_manifest(
    womd_json: str | Path,
    output_path: str | Path,
    release: str = "unknown-supplied-compact-export",
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            build_compact_womd_manifest(womd_json, release=release), indent=2
        ),
        encoding="utf-8",
    )
    return destination
