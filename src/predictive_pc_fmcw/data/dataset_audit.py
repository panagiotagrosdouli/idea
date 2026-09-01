from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_ARRAYS = {
    "history_xy",
    "future_xy",
    "scenario_id",
    "actor_id",
    "future_ego_heading_rad",
    "split",
    "source",
    "coordinate_frame",
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def audit_training_npz(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with np.load(source_path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_ARRAYS.difference(archive.files))
        if missing:
            raise ValueError(f"Training NPZ is missing arrays: {missing}")
        history = archive["history_xy"]
        future = archive["future_xy"]
        headings = archive["future_ego_heading_rad"]
        sample_count = int(history.shape[0])
        expected = {
            "future_xy": sample_count,
            "scenario_id": sample_count,
            "actor_id": sample_count,
            "future_ego_heading_rad": sample_count,
            "split": sample_count,
        }
        for name, count in expected.items():
            if int(archive[name].shape[0]) != count:
                raise ValueError(f"{name} has inconsistent sample count")
        numeric = (history, future, headings)
        finite = all(bool(np.all(np.isfinite(values))) for values in numeric)
        split_names, split_counts = np.unique(
            archive["split"], return_counts=True
        )
        future_displacement = np.linalg.norm(
            future[:, -1] - history[:, -1], axis=1
        )
        history_step = np.linalg.norm(np.diff(history, axis=1), axis=2)
        report = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
            "file_size_bytes": source_path.stat().st_size,
            "sample_count": sample_count,
            "unique_scenarios": int(np.unique(archive["scenario_id"]).size),
            "unique_actor_ids": int(np.unique(archive["actor_id"]).size),
            "history_steps": int(history.shape[1]),
            "future_steps": int(future.shape[1]),
            "coordinate_dimensions": int(history.shape[2]),
            "all_numeric_values_finite": finite,
            "splits": {
                str(name): int(count)
                for name, count in zip(split_names, split_counts, strict=True)
            },
            "source": str(archive["source"].item()),
            "coordinate_frame": str(archive["coordinate_frame"].item()),
            "future_displacement_m": {
                "mean": float(np.mean(future_displacement)),
                "median": float(np.median(future_displacement)),
                "p95": float(np.percentile(future_displacement, 95)),
            },
            "history_step_displacement_m": {
                "mean": float(np.mean(history_step)),
                "median": float(np.median(history_step)),
                "p95": float(np.percentile(history_step, 95)),
            },
        }
    return report


def write_audit_report(report: dict[str, Any], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination
