from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from predictive_pc_fmcw.data.dataset_audit import audit_training_npz


def _scenario_ids(path: str | Path) -> set[str]:
    with np.load(path, allow_pickle=False) as archive:
        if "scenario_id" not in archive.files:
            raise ValueError(f"{path} is missing scenario_id")
        return set(np.asarray(archive["scenario_id"]).astype(str).tolist())


def verify_corpora(
    training_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    training = audit_training_npz(training_path)
    validation = audit_training_npz(validation_path)

    training_ids = _scenario_ids(training_path)
    validation_ids = _scenario_ids(validation_path)
    overlap = sorted(training_ids.intersection(validation_ids))

    shape_checks = {
        "training_history_steps_11": training["history_steps"] == 11,
        "training_future_steps_80": training["future_steps"] == 80,
        "validation_history_steps_11": validation["history_steps"] == 11,
        "validation_future_steps_80": validation["future_steps"] == 80,
    }
    integrity_checks = {
        "training_finite": bool(training["all_numeric_values_finite"]),
        "validation_finite": bool(validation["all_numeric_values_finite"]),
        "training_internal_scenario_integrity": bool(
            training["scenario_split_integrity"]["passed"]
        ),
        "validation_internal_scenario_integrity": bool(
            validation["scenario_split_integrity"]["passed"]
        ),
        "cross_corpus_zero_overlap": not overlap,
    }
    passed = all(shape_checks.values()) and all(integrity_checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "training": training,
        "validation": validation,
        "shape_checks": shape_checks,
        "integrity_checks": integrity_checks,
        "cross_corpus_overlap": {
            "count": len(overlap),
            "scenario_ids": overlap,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of frozen WOMD training and validation corpora."
    )
    parser.add_argument("training_npz")
    parser.add_argument("validation_npz")
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/01_data/corpus_verification.json",
    )
    args = parser.parse_args()

    report = verify_corpora(args.training_npz, args.validation_npz)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
