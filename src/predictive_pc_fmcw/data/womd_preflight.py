from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .corpus_audit import verify_corpora
from .dataset_audit import audit_training_npz


def _discover(roots: Sequence[str | Path]) -> tuple[list[Path], list[Path]]:
    files: set[Path] = set()
    for value in roots:
        root = Path(value)
        if root.is_file():
            files.add(root.resolve())
        elif root.is_dir():
            files.update(path.resolve() for path in root.rglob("*") if path.is_file())
    npz = sorted(path for path in files if path.suffix.lower() == ".npz")
    tfrecords = sorted(
        path for path in files if "tfrecord" in path.name.lower()
    )
    return npz, tfrecords


def preflight_womd_roots(roots: Sequence[str | Path]) -> dict[str, Any]:
    """Find a Stage-1-ready corpus pair without treating presence as validity."""

    npz_paths, tfrecord_paths = _discover(roots)
    candidates: list[dict[str, Any]] = []
    training_paths: list[Path] = []
    validation_paths: list[Path] = []

    for path in npz_paths:
        item: dict[str, Any] = {"path": str(path)}
        try:
            audit = audit_training_npz(path)
            item.update({"valid_npz_schema": True, "audit": audit})
            splits = set(audit["splits"])
            if {"training", "development"}.issubset(splits):
                training_paths.append(path)
            if splits == {"official_validation"}:
                validation_paths.append(path)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            item.update(
                {"valid_npz_schema": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        candidates.append(item)

    pair_checks: list[dict[str, Any]] = []
    selected_pair: dict[str, str] | None = None
    for training_path in training_paths:
        for validation_path in validation_paths:
            check = verify_corpora(training_path, validation_path)
            pair_checks.append(
                {
                    "training_path": str(training_path),
                    "validation_path": str(validation_path),
                    "verification": check,
                }
            )
            if check["status"] == "PASS" and selected_pair is None:
                selected_pair = {
                    "training_path": str(training_path),
                    "validation_path": str(validation_path),
                }

    if selected_pair is not None:
        status = "PASS"
    elif tfrecord_paths:
        status = "BLOCKED_EXPORT_REQUIRED"
    elif npz_paths:
        status = "BLOCKED_INVALID_CORPUS"
    else:
        status = "BLOCKED_NO_DATA"

    return {
        "schema_version": 1,
        "status": status,
        "stage1_ready": status == "PASS",
        "selected_pair": selected_pair,
        "counts": {
            "npz_candidates": len(npz_paths),
            "tfrecord_candidates": len(tfrecord_paths),
            "training_role_candidates": len(training_paths),
            "validation_role_candidates": len(validation_paths),
            "pairs_checked": len(pair_checks),
        },
        "npz_candidates": candidates,
        "tfrecord_candidates": [str(path) for path in tfrecord_paths],
        "pair_checks": pair_checks,
        "interpretation": {
            "PASS": "A corpus pair passed the complete Stage-1 verifier.",
            "BLOCKED_EXPORT_REQUIRED": (
                "Raw TFRecords were found, but frozen training and official-validation "
                "NPZ corpora must still be exported and verified."
            ),
            "BLOCKED_INVALID_CORPUS": (
                "NPZ files were found, but no pair passed schema, provenance, split, "
                "shape, finiteness, and scenario-disjointness checks."
            ),
            "BLOCKED_NO_DATA": "No NPZ or TFRecord candidates were found.",
        }[status],
    }
