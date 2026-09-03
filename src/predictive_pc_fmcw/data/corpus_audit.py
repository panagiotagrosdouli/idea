from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .dataset_audit import audit_training_npz


OFFICIAL_SOURCE = "real_WOMD_v1.3.1_true_SDC_geometry"
OFFICIAL_COORDINATE_FRAME = "world_xy_with_explicit_ego_heading"
OFFICIAL_VALIDATION_SPLIT = "official_validation"


def _scenario_ids(path: str | Path) -> set[str]:
    with np.load(path, allow_pickle=False) as archive:
        if "scenario_id" not in archive.files:
            raise ValueError(f"{path} is missing scenario_id")
        return set(np.asarray(archive["scenario_id"]).astype(str).tolist())


def _split_labels(audit: dict[str, Any]) -> set[str]:
    return set(str(label) for label in audit["split_counts"])


def verify_corpora(
    training_path: str | Path,
    validation_path: str | Path,
) -> dict[str, Any]:
    training = audit_training_npz(training_path)
    validation = audit_training_npz(validation_path)
    training_ids = _scenario_ids(training_path)
    validation_ids = _scenario_ids(validation_path)
    overlap = sorted(training_ids.intersection(validation_ids))
    training_splits = _split_labels(training)
    validation_splits = _split_labels(validation)

    shape_checks = {
        "training_history_steps_11": training["history_steps"] == 11,
        "training_future_steps_80": training["future_steps"] == 80,
        "validation_history_steps_11": validation["history_steps"] == 11,
        "validation_future_steps_80": validation["future_steps"] == 80,
    }
    provenance_checks = {
        "training_official_true_sdc_source": training["source"] == OFFICIAL_SOURCE,
        "validation_official_true_sdc_source": validation["source"] == OFFICIAL_SOURCE,
        "training_coordinate_frame": (
            training["coordinate_frame"] == OFFICIAL_COORDINATE_FRAME
        ),
        "validation_coordinate_frame": (
            validation["coordinate_frame"] == OFFICIAL_COORDINATE_FRAME
        ),
        "training_has_training_and_development": (
            {"training", "development"}.issubset(training_splits)
        ),
        "validation_is_untouched_role": validation_splits
        == {OFFICIAL_VALIDATION_SPLIT},
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
    passed = (
        all(shape_checks.values())
        and all(provenance_checks.values())
        and all(integrity_checks.values())
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "training": training,
        "validation": validation,
        "shape_checks": shape_checks,
        "provenance_checks": provenance_checks,
        "integrity_checks": integrity_checks,
        "cross_corpus_overlap": {"count": len(overlap), "scenario_ids": overlap},
    }
