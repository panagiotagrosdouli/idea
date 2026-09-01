from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import LinkConfig
from ..data.manifest import sha256_file
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
    return tuple(sorted(settings, key=lambda item: (item.lambda_link, item.lambda_outage)))


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
    plan = {
        "dataset": str(source),
        "dataset_sha256": sha256_file(source),
        "seeds": list(seeds),
        "epochs": epochs,
        "batch_size": batch_size,
        "settings": [asdict(setting) for setting in chosen],
        "planned_runs": len(chosen) * len(seeds),
    }
    (destination / "lambda_sweep_plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8"
    )
    results = []
    for setting in chosen:
        for seed in seeds:
            run_dir = destination / setting.name / f"seed_{seed}"
            result_path = run_dir / "training_result.json"
            if result_path.is_file():
                result = TrainingResult(
                    **json.loads(result_path.read_text(encoding="utf-8"))
                )
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
            results.append(result)
            (destination / "lambda_sweep_results.json").write_text(
                json.dumps([asdict(item) for item in results], indent=2) + "\n",
                encoding="utf-8",
            )
    return results
