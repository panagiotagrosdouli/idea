from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STALE_MANUSCRIPT_MARKERS = (
    "reproducible research draft",
    "remain to be inserted before submission",
    "Fifty-one regression tests",
    "In 12 controlled episodes",
    "compact three-scene WOMD proxy",
    "full 1,125-row, five-seed staged study",
    "artifacts/corrected_v2/",
    "remain necessary for submission-quality empirical evidence",
)

CANONICAL_OBJECTIVES = {
    "trajectory_only",
    "trajectory_link",
    "trajectory_outage",
    "full",
}
CANONICAL_SEEDS = {20260827, 20260828, 20260829, 20260830, 20260831}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _semantic_error(path: Path) -> str | None:
    if not path.is_file():
        return None
    if path.stat().st_size == 0:
        return "evidence file is empty"
    if path.suffix != ".json":
        return None

    payload = _load_json(path)
    if payload is None:
        return "invalid JSON evidence"
    name = path.name

    if name == "source_shard_manifest.json":
        training = payload.get("training", {})
        validation = payload.get("validation", {})
        if not (
            payload.get("complete") is True
            and training.get("complete") is True
            and validation.get("complete") is True
        ):
            return "source shard manifest is incomplete"
    elif name in {"corpus_verification.json", "link_verification.json"}:
        if payload.get("status") != "PASS":
            return f"{name} status is not PASS"
    elif name == "completion_manifest.json":
        objectives = set(payload.get("objectives", ()))
        seeds = set(payload.get("seeds", ()))
        checkpoints = payload.get("checkpoints", ())
        if not (
            payload.get("complete") is True
            and payload.get("expected_runs") == 20
            and payload.get("completed_runs") == 20
            and objectives == CANONICAL_OBJECTIVES
            and seeds == CANONICAL_SEEDS
            and len(checkpoints) == 20
            and bool(payload.get("dataset_sha256"))
            and bool(payload.get("link_config_sha256"))
        ):
            return "Stage-4 completion manifest is not canonical 20/20 evidence"
    elif name == "heldout_provenance.json":
        completion = payload.get("completion_verification") or {}
        lut = payload.get("ber_lut_verification") or {}
        link_model = payload.get("link_model") or {}
        if not (
            completion.get("status") == "PASS"
            and lut.get("status") == "PASS"
            and link_model.get("ber_source") == "lut"
        ):
            return "held-out provenance does not verify completion and BER LUT"
    elif name == "evaluation_manifest.json":
        completion = payload.get("completion_verification") or {}
        lut = payload.get("ber_lut_verification") or {}
        if not (
            payload.get("canonical") is True
            and int(payload.get("scenario_count", 0)) > 0
            and payload.get("heldout_scenario_count") == payload.get("scenario_count")
            and len(payload.get("checkpoints", ())) == 20
            and len(payload.get("paired_schedulers", ())) == 8
            and set(payload.get("traffic_seeds", ())) == CANONICAL_SEEDS
            and payload.get("independent_statistical_unit") == "scenario_id"
            and completion.get("status") == "PASS"
            and lut.get("status") == "PASS"
        ):
            return "Stage-6 evaluation manifest is not canonical paired evidence"
    elif name == "learned_heldout_analysis.json":
        correlation = payload.get("accuracy_link_correlations") or {}
        comparison = payload.get("full_vs_trajectory_only") or {}
        objectives = payload.get("objective_summary") or {}
        if not (
            int(correlation.get("independent_scenarios", 0)) > 1
            and int(comparison.get("pairs", 0)) > 0
            and set(objectives) == CANONICAL_OBJECTIVES
        ):
            return "learned held-out analysis lacks canonical scenario evidence"
    elif name == "scheduler_utility_summary.json":
        relationship = payload.get("ade_vs_realized_goodput_gain") or {}
        objective_keys = set(payload).difference({"ade_vs_realized_goodput_gain"})
        if not (
            int(relationship.get("independent_scenarios", 0)) > 1
            and objective_keys == CANONICAL_OBJECTIVES
        ):
            return "scheduler utility summary lacks canonical scenario evidence"
    return None


def verify_release_readiness(
    manuscript_path: str | Path,
    required_evidence: list[str | Path],
) -> dict[str, Any]:
    manuscript = Path(manuscript_path)
    if not manuscript.is_file():
        return {
            "status": "BLOCKED",
            "manuscript": str(manuscript),
            "checks": {"manuscript_exists": False},
            "stale_markers": [],
            "missing_evidence": [],
            "invalid_evidence": [],
            "required_evidence": [str(path) for path in required_evidence],
        }

    text = manuscript.read_text(encoding="utf-8")
    evidence = [Path(path) for path in required_evidence]
    stale = [marker for marker in STALE_MANUSCRIPT_MARKERS if marker in text]
    missing = [str(path) for path in evidence if not path.is_file()]
    invalid = []
    for path in evidence:
        error = _semantic_error(path)
        if error is not None:
            invalid.append({"path": str(path), "reason": error})
    checks = {
        "manuscript_exists": True,
        "no_stale_draft_markers": not stale,
        "all_required_official_evidence_exists": not missing,
        "all_required_evidence_is_semantically_valid": not invalid,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "BLOCKED",
        "manuscript": str(manuscript),
        "checks": checks,
        "stale_markers": stale,
        "missing_evidence": missing,
        "invalid_evidence": invalid,
        "required_evidence": [str(path) for path in evidence],
    }
