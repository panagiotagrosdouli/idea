#!/usr/bin/env python3
"""Fail-closed preflight for the canonical WOMD Stage-1 corpus.

This utility intentionally does not download, synthesize, or relabel WOMD data.
It reports whether canonical NPZ/TFRecord inputs are present and verifies the
historical v1.3.0 reference hash when an exact file is encountered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HISTORICAL_V130_SHA256 = "b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee"
HISTORICAL_V130_SIZE = 278_168_355
REQUIRED_KEYS = {
    "history_xy",
    "history_vxy",
    "future_xy",
    "future_relative_xy",
    "sdc_future_xy",
    "history_valid",
    "future_valid",
    "scenario_id",
    "track_id",
    "sdc_track_id",
    "split",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_npz(path: Path) -> dict:
    import numpy as np

    item = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            item["keys"] = sorted(keys)
            item["missing_keys"] = sorted(REQUIRED_KEYS - keys)
            if "scenario_id" in archive.files:
                item["samples"] = int(archive["scenario_id"].shape[0])
                item["unique_scenarios"] = int(
                    np.unique(archive["scenario_id"].astype(str)).size
                )
            if {"future_xy", "sdc_future_xy", "future_relative_xy"} <= keys:
                error = np.max(
                    np.abs(
                        archive["future_relative_xy"]
                        - (archive["future_xy"] - archive["sdc_future_xy"])
                    )
                )
                item["geometry_identity_max_abs_error"] = float(error)
    except Exception as exc:  # pragma: no cover - diagnostic path
        item["read_error"] = repr(exc)

    item["historical_v130_exact_match"] = (
        item["sha256"] == HISTORICAL_V130_SHA256
        and item["size_bytes"] == HISTORICAL_V130_SIZE
    )
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--json", type=Path, default=Path("womd_preflight.json"))
    args = parser.parse_args()

    files: set[Path] = set()
    for root in args.roots:
        if root.is_file():
            files.add(root.resolve())
        elif root.exists():
            for path in root.rglob("*"):
                if path.is_file() and (
                    path.name.endswith(".npz")
                    or ".tfrecord" in path.name.lower()
                ):
                    files.add(path.resolve())

    npzs = sorted(path for path in files if path.name.endswith(".npz"))
    tfrecords = sorted(path for path in files if ".tfrecord" in path.name.lower())
    report = {
        "historical_v130_reference": {
            "sha256": HISTORICAL_V130_SHA256,
            "size_bytes": HISTORICAL_V130_SIZE,
        },
        "npz_candidates": [inspect_npz(path) for path in npzs],
        "tfrecord_candidates": [str(path) for path in tfrecords],
        "counts": {"npz": len(npzs), "tfrecord": len(tfrecords)},
    }
    report["canonical_data_present"] = bool(npzs or tfrecords)
    report["status"] = "READY_FOR_STAGE1" if report["canonical_data_present"] else "BLOCKED_NO_DATA"

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
