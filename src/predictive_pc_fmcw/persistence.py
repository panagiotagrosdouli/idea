from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PersistenceValidation:
    valid: bool
    reason: str
    path: str
    size_bytes: int


def validate_completed_file(
    path: str | Path,
    *,
    minimum_bytes: int = 1,
    expected_size: int | None = None,
) -> PersistenceValidation:
    source = Path(path)
    if source.name.endswith((".part", ".persisting")):
        return PersistenceValidation(False, "temporary_file", str(source), 0)
    if not source.is_file():
        return PersistenceValidation(False, "missing", str(source), 0)
    size = source.stat().st_size
    if size < minimum_bytes:
        return PersistenceValidation(False, "too_small", str(source), size)
    if expected_size is not None and size != expected_size:
        return PersistenceValidation(False, "size_mismatch", str(source), size)
    return PersistenceValidation(True, "complete", str(source), size)


def remove_stale_persistence_files(final_path: str | Path) -> None:
    final = Path(final_path)
    final.with_name(final.name + ".persisting").unlink(missing_ok=True)


def persist_completed_file_atomic(
    local_completed: str | Path,
    final_path: str | Path,
    *,
    expected_size: int | None = None,
) -> PersistenceValidation:
    source = Path(local_completed)
    if not source.is_file() or source.stat().st_size < 1:
        raise ValueError(f"Completed local file is missing or empty: {source}")
    if expected_size is not None and source.stat().st_size != expected_size:
        raise ValueError(
            f"Completed local file has {source.stat().st_size} bytes; expected {expected_size}."
        )

    final = Path(final_path)
    final.parent.mkdir(parents=True, exist_ok=True)
    persisting = final.with_name(final.name + ".persisting")
    persisting.unlink(missing_ok=True)
    shutil.copy2(source, persisting)

    if not persisting.is_file() or persisting.stat().st_size < 1:
        persisting.unlink(missing_ok=True)
        raise RuntimeError(f"Persistence copy failed validation: {persisting}")
    if expected_size is not None and persisting.stat().st_size != expected_size:
        actual = persisting.stat().st_size
        persisting.unlink(missing_ok=True)
        raise RuntimeError(
            f"Persistence copy has {actual} bytes; expected {expected_size}."
        )

    os.replace(persisting, final)
    return validate_completed_file(final, expected_size=expected_size)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
