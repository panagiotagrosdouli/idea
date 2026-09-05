from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import LinkConfig
from ..data.manifest import sha256_file
from ..persistence import atomic_write_json
from .ablation import validate_training_resume
from .train import TrainingResult, train_from_npz


@dataclass(frozen=True)
class LambdaSetting:
    lambda_link: float
    lambda_outage: float

    @property
    def name(self) -> str:
        link = str(self.lambda_link).replace(".", "p")
        outage = str(self.lambda_outage).replace(".", "p")
        return f"link_{link}_outage_{outage}"


def default_lambda_settings() -> tuple[LambdaSetting, ...]:
    settings = {
        LambdaSetting(value, 0.1) for value in (0.05, 0.2, 0.5)
    } | {
        LambdaSetting(0.2, value) for value in (0.05, 0.1, 0.2)
    }
    return tuple(
        sorted(
            settings,
            key=lambda item: (item.lambda_link, item.lambda_outage),
        )
    )


def run_lambda_sweep(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    link_config: LinkConfig,
    seeds: tuple[int, ...],
    epochs: int,
    batch_size: int = 32,
    settings: tuple[LambdaSetting, ...] | None = None,
) -> list[TrainingResult]:
    if len(seeds) < 3:
        raise ValueError("Publication lambda sweep requires at least three seeds.")
    source = Path(dataset_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    chosen = settings or default_lambda_settings()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_sha256 = sha256_file(source)
    plan = {
        "dataset": str(source),
        "dataset_sha256": dataset_sha256,
        "selection_scope": "development_only",
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "settings": [asdict(setting) for setting in chosen],
        "planned_runs": len(chosen) * len(seeds),
    }
    atomic_write_json(destination / "lambda_sweep_plan.json", plan)
    results: list[TrainingResult] = []
    execution_runs: list[dict[str, object]] = []
    for setting in chosen:
        for seed in seeds:
            run_dir = destination / setting.name / f"seed_{seed}"
            result_path = run_dir / "training_result.json"
            validation = validate_training_resume(
                result_path,
                expected_objective="full",
                expected_seed=seed,
                expected_dataset_sha256=dataset_sha256,
                expected_run_dir=run_dir,
            )
            if validation.valid and validation.result is not None:
                result = validation.result
                reason = validation.reason
            else:
                result = train_from_npz(
                    source,
                    run_dir,
                    epochs=epochs,
                    batch_size=batch_size,
                    lambda_link=setting.lambda_link,
                    lambda_outage=setting.lambda_outage,
                    objective="full",
                    link_config=link_config,
                    seed=seed,
                )
                post_validation = validate_training_resume(
                    result_path,
                    expected_objective="full",
                    expected_seed=seed,
                    expected_dataset_sha256=dataset_sha256,
                    expected_run_dir=run_dir,
                )
                if not post_validation.valid or post_validation.result is None:
                    message = (
                        f"Lambda sweep run {setting.name} seed {seed} "
                        "failed validation: "
                        f"{post_validation.reason}"
                    )
                    raise RuntimeError(message)
                result = post_validation.result
                reason = f"rerun_after:{validation.reason}"
            results.append(result)
            execution_runs.append(
                {
                    "setting": asdict(setting),
                    "seed": seed,
                    "status": "completed",
                    "result_path": str(result_path),
                    "checkpoint": result.checkpoint,
                    "validation_reason": reason,
                }
            )
            atomic_write_json(
                destination / "lambda_sweep_results.json",
                [asdict(item) for item in results],
            )
            atomic_write_json(
                destination / "execution_state.json",
                {
                    "schema": "lambda_sweep_execution_state_v1",
                    "operational_only": True,
                    "selection_scope": "development_only",
                    "status": (
                        "completed"
                        if len(execution_runs) == plan["planned_runs"]
                        else "in_progress"
                    ),
                    "planned_runs": plan["planned_runs"],
                    "completed_runs": len(execution_runs),
                    "runs": execution_runs,
                },
            )
    return results
