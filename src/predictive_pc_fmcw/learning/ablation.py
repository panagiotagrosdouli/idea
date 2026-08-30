from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import LinkConfig
from ..data.manifest import sha256_file
from .train import TrainingResult, train_from_npz

OBJECTIVES = (
    "trajectory_only",
    "trajectory_link",
    "trajectory_outage",
    "full",
)


@dataclass(frozen=True)
class TrainingAblationPlan:
    dataset: str
    dataset_sha256: str
    objectives: tuple[str, ...]
    seeds: tuple[int, ...]
    epochs: int
    planned_runs: int


def build_training_ablation_plan(
    dataset_path: str | Path,
    seeds: tuple[int, ...],
    epochs: int,
) -> TrainingAblationPlan:
    source = Path(dataset_path)
    if not source.is_file():
        raise FileNotFoundError(f"Training dataset not found: {source}")
    if len(seeds) < 3:
        raise ValueError("Publication ablation requires at least three seeds.")
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


def run_training_ablation(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    link_config: LinkConfig,
    seeds: tuple[int, ...] = (20260827, 20260828, 20260829),
    epochs: int = 80,
    batch_size: int = 64,
    lambda_link: float = 0.2,
    lambda_outage: float = 0.1,
) -> list[TrainingResult]:
    plan = build_training_ablation_plan(dataset_path, seeds, epochs)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ablation_plan.json").write_text(
        json.dumps(asdict(plan), indent=2), encoding="utf-8"
    )
    results: list[TrainingResult] = []
    for objective in OBJECTIVES:
        for seed in seeds:
            run_dir = destination / objective / f"seed_{seed}"
            completed_result = run_dir / "training_result.json"
            if completed_result.is_file():
                payload = json.loads(completed_result.read_text(encoding="utf-8"))
                result = TrainingResult(**payload)
            else:
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
            results.append(result)
            (destination / "ablation_results.json").write_text(
                json.dumps([asdict(item) for item in results], indent=2),
                encoding="utf-8",
            )
    return results
