from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import LinkConfig
from ..data.manifest import sha256_file
from ..persistence import atomic_write_json, validate_completed_file
from .train import TrainingResult, train_from_npz

OBJECTIVES = (
    "trajectory_only",
    "trajectory_link",
    "trajectory_outage",
    "full",
)
CANONICAL_SEEDS = (20260827, 20260828, 20260829, 20260830, 20260831)


@dataclass(frozen=True)
class TrainingAblationPlan:
    dataset: str
    dataset_sha256: str
    objectives: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: int
    planned_runs: int


@dataclass(frozen=True)
class ResumeValidation:
    valid: bool
    reason: str
    result: TrainingResult | None


def _link_config_sha256(link_config: LinkConfig) -> str:
    payload = json.dumps(
        asdict(link_config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_checkpoint(checkpoint_value: str, run_dir: Path) -> Path:
    checkpoint = Path(checkpoint_value)
    if checkpoint.is_absolute():
        return checkpoint.resolve()
    cwd_candidate = (Path.cwd() / checkpoint).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (run_dir / checkpoint).resolve()


def validate_training_resume(
    result_path: str | Path,
    *,
    expected_objective: str,
    expected_seed: int,
    expected_dataset_sha256: str,
    expected_run_dir: str | Path,
) -> ResumeValidation:
    source = Path(result_path)
    if not source.is_file():
        return ResumeValidation(False, "missing_result", None)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("training result must be a JSON object")
        result = TrainingResult(**payload)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        reason = f"invalid_result_json:{type(exc).__name__}"
        return ResumeValidation(False, reason, None)

    if result.objective != expected_objective:
        return ResumeValidation(False, "objective_mismatch", None)
    if result.seed != expected_seed:
        return ResumeValidation(False, "seed_mismatch", None)
    if result.dataset_sha256 != expected_dataset_sha256:
        return ResumeValidation(False, "dataset_hash_mismatch", None)

    run_dir = Path(expected_run_dir).resolve()
    checkpoint = _resolve_checkpoint(result.checkpoint, run_dir)
    try:
        checkpoint.relative_to(run_dir)
    except ValueError:
        return ResumeValidation(False, "checkpoint_outside_run_dir", None)

    checkpoint_validation = validate_completed_file(checkpoint)
    if not checkpoint_validation.valid:
        return ResumeValidation(
            False, f"checkpoint_{checkpoint_validation.reason}", None
        )
    return ResumeValidation(True, "verified_complete", result)


def build_training_ablation_plan(
    dataset_path: str | Path,
    seeds: tuple[int, ...],
    epochs: int,
) -> TrainingAblationPlan:
    source = Path(dataset_path)
    if not source.is_file():
        raise FileNotFoundError(f"Training dataset not found: {source}")
    if seeds != CANONICAL_SEEDS:
        raise ValueError(
            "Publication ablation requires exactly the five frozen seeds: "
            f"{CANONICAL_SEEDS}."
        )
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    return TrainingAblationPlan(
        dataset=str(source),
        dataset_sha256=sha256_file(source),
        objectives=OBJECTIVES,
        seeds=seeds,
        epochs=epochs,
        planned_runs=len(OBJECTIVES) * len(seeds),
    )


def _write_execution_state(
    destination: Path,
    plan: TrainingAblationPlan,
    runs: list[dict[str, Any]],
) -> None:
    statuses = [str(item["status"]) for item in runs]
    completed = sum(status == "completed" for status in statuses)
    state = {
        "schema": "stage4_execution_state_v1",
        "operational_only": True,
        "scientific_completion_gate": "completion_manifest.json",
        "stage": "stage4",
        "status": (
            "completed"
            if completed == plan.planned_runs
            else "in_progress"
        ),
        "planned_runs": plan.planned_runs,
        "completed_runs": completed,
        "pending_runs": plan.planned_runs - completed,
        "invalid_runs": sum(status == "invalid" for status in statuses),
        "rerun_required_runs": sum(status == "rerun_required" for status in statuses),
        "dataset": plan.dataset,
        "dataset_sha256": plan.dataset_sha256,
        "runs": runs,
    }
    atomic_write_json(destination / "execution_state.json", state)


def run_training_ablation(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    link_config: LinkConfig,
    seeds: tuple[int, ...] = CANONICAL_SEEDS,
    epochs: int = 80,
    batch_size: int = 32,
    lambda_link: float = 0.2,
    lambda_outage: float = 0.1,
) -> list[TrainingResult]:
    plan = build_training_ablation_plan(dataset_path, seeds, epochs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "ablation_plan.json", asdict(plan))

    results: list[TrainingResult] = []
    run_states: list[dict[str, Any]] = []
    verified_checkpoints: list[Path] = []
    _write_execution_state(destination, plan, run_states)

    for objective in OBJECTIVES:
        for seed in seeds:
            run_dir = destination / objective / f"seed_{seed}"
            completed_result = run_dir / "training_result.json"
            validation = validate_training_resume(
                completed_result,
                expected_objective=objective,
                expected_seed=seed,
                expected_dataset_sha256=plan.dataset_sha256,
                expected_run_dir=run_dir,
            )
            if validation.valid and validation.result is not None:
                result = validation.result
                reason = validation.reason
            else:
                run_states.append(
                    {
                        "objective": objective,
                        "seed": seed,
                        "status": "rerun_required",
                        "result_path": str(completed_result),
                        "checkpoint": None,
                        "validation_reason": validation.reason,
                    }
                )
                _write_execution_state(destination, plan, run_states)
                result = train_from_npz(
                    dataset_path,
                    run_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lambda_link=lambda_link,
                    lambda_outage=lambda_outage,
                    objective=objective,
                    link_config=link_config,
                    seed=seed,
                )
                post_validation = validate_training_resume(
                    completed_result,
                    expected_objective=objective,
                    expected_seed=seed,
                    expected_dataset_sha256=plan.dataset_sha256,
                    expected_run_dir=run_dir,
                )
                if not post_validation.valid or post_validation.result is None:
                    run_states[-1].update(
                        {
                            "status": "invalid",
                            "checkpoint": result.checkpoint,
                            "validation_reason": post_validation.reason,
                        }
                    )
                    _write_execution_state(destination, plan, run_states)
                    message = (
                        f"Stage 4 run {objective} seed {seed} failed "
                        "completion validation: "
                        f"{post_validation.reason}"
                    )
                    raise RuntimeError(message)
                run_states.pop()
                result = post_validation.result
                reason = f"rerun_after:{validation.reason}"

            checkpoint = _resolve_checkpoint(result.checkpoint, run_dir.resolve())
            results.append(result)
            verified_checkpoints.append(checkpoint)
            run_states.append(
                {
                    "objective": objective,
                    "seed": seed,
                    "status": "completed",
                    "result_path": str(completed_result),
                    "checkpoint": str(checkpoint),
                    "validation_reason": reason,
                }
            )
            atomic_write_json(
                destination / "ablation_results.json",
                [asdict(item) for item in results],
            )
            _write_execution_state(destination, plan, run_states)

    expected = len(OBJECTIVES) * len(seeds)
    if len(results) == expected and all(
        validate_completed_file(path).valid for path in verified_checkpoints
    ):
        completion = {
            "complete": True,
            "completed_runs": len(results),
            "expected_runs": expected,
            "dataset_sha256": plan.dataset_sha256,
            "link_config": asdict(link_config),
            "link_config_sha256": _link_config_sha256(link_config),
            "objectives": list(OBJECTIVES),
            "seeds": list(seeds),
            "epochs": epochs,
            "batch_size": batch_size,
            "lambda_link": lambda_link,
            "lambda_outage": lambda_outage,
            "checkpoints": [str(path) for path in verified_checkpoints],
        }
        atomic_write_json(destination / "completion_manifest.json", completion)
    return results
