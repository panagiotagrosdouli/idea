from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    InteractingMultipleModelPredictor,
    KalmanConstantVelocityPredictor,
    LastPositionPredictor,
)
from .scheduling.base import SchedulerContext
from .scheduling.policies import build_scheduler


@dataclass(frozen=True)
class ComplexityRow:
    component: str
    kind: str
    parameters: int
    median_runtime_us: float
    p95_runtime_us: float
    repeats: int
    entities: int
    horizon_steps: int


def _runtime_samples(function, repeats: int, warmup: int) -> np.ndarray:
    for _ in range(warmup):
        function()
    samples = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        started = time.perf_counter_ns()
        function()
        samples[index] = (time.perf_counter_ns() - started) / 1_000.0
    return samples


def _gru_parameter_count(horizon_steps: int, hidden: int = 128, layers: int = 2) -> int:
    count = 0
    input_size = 4
    for layer in range(layers):
        layer_input = input_size if layer == 0 else hidden
        count += 3 * hidden * layer_input
        count += 3 * hidden * hidden
        count += 6 * hidden
    count += hidden * hidden + hidden
    count += 2 * horizon_steps * hidden + 2 * horizon_steps
    return count


def measure_complexity(
    config: ExperimentConfig,
    *,
    repeats: int = 200,
    warmup: int = 20,
) -> list[ComplexityRow]:
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup non-negative.")
    entities = config.benchmark.vehicles + 1
    rng = np.random.default_rng(config.seed)
    history = np.cumsum(rng.normal(size=(entities, 10, 2)), axis=1)
    predictors = (
        LastPositionPredictor(),
        ConstantVelocityPredictor(),
        ConstantAccelerationPredictor(),
        KalmanConstantVelocityPredictor(),
        InteractingMultipleModelPredictor(),
    )
    rows: list[ComplexityRow] = []
    for predictor in predictors:
        samples = _runtime_samples(
            lambda current=predictor: current.predict(
                history,
                config.prediction_horizon_steps,
                config.slot_duration_s,
            ),
            repeats,
            warmup,
        )
        rows.append(
            ComplexityRow(
                component=predictor.name,
                kind="trajectory_predictor",
                parameters=0,
                median_runtime_us=float(np.median(samples)),
                p95_runtime_us=float(np.quantile(samples, 0.95)),
                repeats=repeats,
                entities=entities,
                horizon_steps=config.prediction_horizon_steps,
            )
        )

    vehicles = config.benchmark.vehicles
    horizon = config.prediction_horizon_steps
    context = SchedulerContext(
        slot=20,
        queue_lengths=np.arange(1, vehicles + 1, dtype=np.int64),
        time_to_deadline=np.linspace(1.0, 6.0, vehicles),
        current_goodput_bps=np.linspace(4e8, 9e8, vehicles),
        current_outage=np.zeros(vehicles, dtype=bool),
        predicted_goodput_bps=rng.uniform(2e8, 1e9, size=(vehicles, horizon)),
        predicted_outage=rng.random((vehicles, horizon)) < 0.15,
        predicted_lifetime_steps=rng.integers(0, horizon + 1, size=vehicles),
        delivered_bits=np.linspace(1e6, 5e6, vehicles),
        previous_vehicle=0,
        data_rate_bps=config.link.data_rate_bps,
        discount=config.discount,
        oracle_forecast=True,
    )
    for name in config.benchmark.schedulers:
        scheduler = build_scheduler(name, config.scheduler, config.seed)
        samples = _runtime_samples(
            lambda current=scheduler: current.select(context),
            repeats,
            warmup,
        )
        rows.append(
            ComplexityRow(
                component=name,
                kind="scheduler",
                parameters=0,
                median_runtime_us=float(np.median(samples)),
                p95_runtime_us=float(np.quantile(samples, 0.95)),
                repeats=repeats,
                entities=vehicles,
                horizon_steps=horizon,
            )
        )
    rows.append(
        ComplexityRow(
            component="gru_128x2_analytical_count",
            kind="learned_predictor_unexecuted",
            parameters=_gru_parameter_count(horizon),
            median_runtime_us=float("nan"),
            p95_runtime_us=float("nan"),
            repeats=0,
            entities=entities,
            horizon_steps=horizon,
        )
    )
    return rows


def write_complexity_artifacts(
    rows: list[ComplexityRow], output_dir: str | Path
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No complexity rows were produced.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    values = [asdict(row) for row in rows]
    json_path = destination / "complexity.json"
    json_path.write_text(
        json.dumps(
            {
                "measurement_scope": "single-process CPU wall-clock diagnostic",
                "platform": platform.platform(),
                "python": platform.python_version(),
                "rows": values,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    csv_path = destination / "complexity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=values[0].keys())
        writer.writeheader()
        writer.writerows(values)
    tex_path = destination / "complexity.tex"
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Component & Kind & Parameters & Median $\mu$s & P95 $\mu$s \\",
        r"\midrule",
    ]
    for row in rows:
        name = row.component.replace("_", r"\_")
        kind = row.kind.replace("_", r"\_")
        lines.append(
            f"{name} & {kind} & {row.parameters} & "
            f"{row.median_runtime_us:.2f} & {row.p95_runtime_us:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "tex": tex_path}
