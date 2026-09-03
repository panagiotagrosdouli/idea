from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data.manifest import sha256_file
from .ablation import CANONICAL_SEEDS


def freeze_lambda_selection(
    sweep_dir: str | Path,
    dataset_path: str | Path,
    *,
    lambda_link: float,
    lambda_outage: float,
    rationale: str,
) -> dict[str, Any]:
    root = Path(sweep_dir)
    plan_path = root / "lambda_sweep_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    dataset_hash = sha256_file(dataset_path)
    settings = {
        (float(item["lambda_link"]), float(item["lambda_outage"]))
        for item in plan.get("settings", [])
    }
    selected = (float(lambda_link), float(lambda_outage))
    expected_runs = len(settings) * len(CANONICAL_SEEDS)
    completed = list(root.glob("*/seed_*/training_result.json"))
    checks = {
        "training_dataset_hash_matches": plan.get("dataset_sha256") == dataset_hash,
        "sweep_uses_frozen_five_seeds": tuple(plan.get("seeds", ()))
        == CANONICAL_SEEDS,
        "selected_pair_was_swept": selected in settings,
        "sweep_complete": len(completed) == expected_runs,
        "rationale_recorded": bool(rationale.strip()),
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "selection_scope": "development_only",
        "lambda_link": selected[0],
        "lambda_outage": selected[1],
        "rationale": rationale.strip(),
        "dataset_sha256": dataset_hash,
        "sweep_plan": str(plan_path),
        "completed_sweep_runs": len(completed),
        "expected_sweep_runs": expected_runs,
        "checks": checks,
    }


def load_lambda_selection(
    selection_path: str | Path,
    dataset_path: str | Path,
) -> tuple[float, float]:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError("Lambda selection artifact is not PASS.")
    if payload.get("selection_scope") != "development_only":
        raise ValueError("Lambda selection must be development-only.")
    if payload.get("dataset_sha256") != sha256_file(dataset_path):
        raise ValueError("Lambda selection dataset hash does not match training corpus.")
    return float(payload["lambda_link"]), float(payload["lambda_outage"])
