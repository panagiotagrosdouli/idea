from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..data.manifest import sha256_file
from .ablation import CANONICAL_SEEDS, validate_training_resume
from .lambda_sweep import LambdaSetting


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


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
    plan = _load_json(plan_path)
    dataset_hash = sha256_file(dataset_path)
    settings = {
        (float(item["lambda_link"]), float(item["lambda_outage"]))
        for item in plan.get("settings", [])
    }
    selected = (float(lambda_link), float(lambda_outage))
    seeds = tuple(plan.get("seeds", ()))
    expected_runs = len(settings) * len(CANONICAL_SEEDS)

    validation_rows: list[dict[str, Any]] = []
    valid_runs = 0
    if seeds == CANONICAL_SEEDS and plan.get("dataset_sha256") == dataset_hash:
        for link_value, outage_value in sorted(settings):
            setting = LambdaSetting(link_value, outage_value)
            for seed in CANONICAL_SEEDS:
                run_dir = root / setting.name / f"seed_{seed}"
                result_path = run_dir / "training_result.json"
                validation = validate_training_resume(
                    result_path,
                    expected_objective="full",
                    expected_seed=seed,
                    expected_dataset_sha256=dataset_hash,
                    expected_run_dir=run_dir,
                )
                if validation.valid:
                    valid_runs += 1
                validation_rows.append(
                    {
                        "setting": {
                            "lambda_link": link_value,
                            "lambda_outage": outage_value,
                        },
                        "seed": seed,
                        "valid": validation.valid,
                        "reason": validation.reason,
                        "result_path": str(result_path),
                    }
                )

    checks = {
        "selection_scope_development_only": plan.get("selection_scope")
        in (None, "development_only"),
        "training_dataset_hash_matches": plan.get("dataset_sha256") == dataset_hash,
        "sweep_uses_frozen_five_seeds": seeds == CANONICAL_SEEDS,
        "selected_pair_was_swept": selected in settings,
        "sweep_complete": valid_runs == expected_runs,
        "rationale_recorded": bool(rationale.strip()),
        "selected_values_finite": all(math.isfinite(value) for value in selected),
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
        "completed_sweep_runs": valid_runs,
        "expected_sweep_runs": expected_runs,
        "checks": checks,
        "run_validation": validation_rows,
    }


def validate_lambda_selection(
    selection_path: str | Path,
    dataset_path: str | Path,
) -> dict[str, Any]:
    source = Path(selection_path)
    try:
        payload = _load_json(source)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {
            "valid": False,
            "reason": f"invalid_selection_json:{type(exc).__name__}",
            "path": str(source),
        }

    dataset_hash = sha256_file(dataset_path)
    if payload.get("status") != "PASS":
        return {"valid": False, "reason": "selection_not_pass", "path": str(source)}
    if payload.get("selection_scope") != "development_only":
        return {
            "valid": False,
            "reason": "selection_not_development_only",
            "path": str(source),
        }
    if payload.get("dataset_sha256") != dataset_hash:
        return {
            "valid": False,
            "reason": "selection_dataset_hash_mismatch",
            "path": str(source),
        }
    checks = payload.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        return {
            "valid": False,
            "reason": "selection_checks_not_all_pass",
            "path": str(source),
        }
    try:
        selected = (float(payload["lambda_link"]), float(payload["lambda_outage"]))
    except (KeyError, TypeError, ValueError):
        return {
            "valid": False,
            "reason": "selection_values_invalid",
            "path": str(source),
        }
    if not all(math.isfinite(value) for value in selected):
        return {
            "valid": False,
            "reason": "selection_values_nonfinite",
            "path": str(source),
        }
    return {
        "valid": True,
        "reason": "verified_development_only_selection",
        "path": str(source),
        "lambda_link": selected[0],
        "lambda_outage": selected[1],
    }


def load_lambda_selection(
    selection_path: str | Path,
    dataset_path: str | Path,
) -> tuple[float, float]:
    validation = validate_lambda_selection(selection_path, dataset_path)
    if not validation["valid"]:
        raise ValueError(f"Invalid lambda selection: {validation['reason']}")
    return float(validation["lambda_link"]), float(validation["lambda_outage"])
