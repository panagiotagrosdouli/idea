from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data.scenario import MotionScenario
from .geometry import heading_from_positions, range_and_bearing
from .link import LinkModel
from .predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    InteractingMultipleModelPredictor,
    KalmanConstantVelocityPredictor,
    LastPositionPredictor,
    TrajectoryPredictor,
    forecast_scenario,
)


@dataclass(frozen=True)
class ForecastMetricRow:
    predictor: str
    scenario_id: str
    actor_id: str
    anchor_index: int
    horizon_steps: int
    ade_m: float
    fde_m: float
    range_mae_m: float
    snr_mae_db: float
    outage_f1: float
    outage_auroc: float
    link_lifetime_error_steps: float


def default_predictors() -> dict[str, TrajectoryPredictor | None]:
    return {
        "last_position": LastPositionPredictor(),
        "constant_velocity": ConstantVelocityPredictor(),
        "kalman_cv": KalmanConstantVelocityPredictor(),
        "imm": InteractingMultipleModelPredictor(),
        "constant_acceleration": ConstantAccelerationPredictor(),
        "oracle": None,
    }


def evaluate_motion_and_link_forecasts(
    scenarios: Iterable[MotionScenario],
    link_model: LinkModel,
    horizon_steps: int,
    predictors: Mapping[str, TrajectoryPredictor | None] | None = None,
    anchor_stride: int = 5,
) -> list[ForecastMetricRow]:
    if horizon_steps < 1 or anchor_stride < 1:
        raise ValueError("horizon_steps and anchor_stride must be positive.")
    predictor_map = dict(predictors or default_predictors())
    rows: list[ForecastMetricRow] = []
    for scenario in scenarios:
        combined = scenario.combined_positions()
        first_anchor = max(2, scenario.start_index - 1)
        for anchor in range(first_anchor, combined.shape[0] - 1, anchor_stride):
            horizon = min(horizon_steps, combined.shape[0] - anchor - 1)
            indices = np.arange(anchor + 1, anchor + horizon + 1)
            actual = combined[indices].transpose(1, 0, 2)
            for name, predictor in predictor_map.items():
                bundle = forecast_scenario(
                    combined,
                    anchor,
                    horizon,
                    scenario.dt_s,
                    predictor,
                    oracle=name == "oracle",
                )
                predicted = np.concatenate(
                    [bundle.ego_xy[None, :, :], bundle.vehicle_xy], axis=0
                )
                rows.extend(
                    _rows_for_forecast(
                        scenario,
                        anchor,
                        name,
                        predicted,
                        actual,
                        link_model,
                    )
                )
    return rows


def _rows_for_forecast(
    scenario: MotionScenario,
    anchor: int,
    predictor_name: str,
    predicted: np.ndarray,
    actual: np.ndarray,
    link_model: LinkModel,
) -> list[ForecastMetricRow]:
    predicted_relative = predicted[1:] - predicted[:1]
    actual_relative = actual[1:] - actual[:1]
    displacement = np.linalg.norm(predicted_relative - actual_relative, axis=-1)
    predicted_heading = heading_from_positions(
        np.concatenate(
            [scenario.ego_positions_xy[anchor, None], predicted[0]], axis=0
        )
    )[1:]
    actual_heading = heading_from_positions(
        np.concatenate(
            [scenario.ego_positions_xy[anchor, None], actual[0]], axis=0
        )
    )[1:]
    predicted_range, predicted_bearing = range_and_bearing(
        predicted[1:], predicted[:1], predicted_heading[None, :]
    )
    actual_range, actual_bearing = range_and_bearing(
        actual[1:], actual[:1], actual_heading[None, :]
    )
    predicted_link = link_model.evaluate_arrays(
        predicted_range, predicted_bearing
    )
    actual_link = link_model.evaluate_arrays(actual_range, actual_bearing)
    predicted_lifetime = link_model.link_lifetime_steps(
        predicted_range, predicted_bearing
    )
    actual_lifetime = link_model.link_lifetime_steps(
        actual_range, actual_bearing
    )
    rows: list[ForecastMetricRow] = []
    for vehicle, actor_id in enumerate(scenario.actor_ids):
        true_outage = actual_link["outage"][vehicle].astype(bool)
        predicted_outage = predicted_link["outage"][vehicle].astype(bool)
        outage_score = -predicted_link["snr_db"][vehicle]
        rows.append(
            ForecastMetricRow(
                predictor=predictor_name,
                scenario_id=scenario.scenario_id,
                actor_id=actor_id,
                anchor_index=anchor,
                horizon_steps=predicted.shape[1],
                ade_m=float(displacement[vehicle].mean()),
                fde_m=float(displacement[vehicle, -1]),
                range_mae_m=float(
                    np.abs(
                        predicted_range[vehicle] - actual_range[vehicle]
                    ).mean()
                ),
                snr_mae_db=float(
                    np.abs(
                        predicted_link["snr_db"][vehicle]
                        - actual_link["snr_db"][vehicle]
                    ).mean()
                ),
                outage_f1=_binary_f1(true_outage, predicted_outage),
                outage_auroc=_binary_auroc(true_outage, outage_score),
                link_lifetime_error_steps=float(
                    abs(predicted_lifetime[vehicle] - actual_lifetime[vehicle])
                ),
            )
        )
    return rows


def _binary_f1(labels: np.ndarray, predictions: np.ndarray) -> float:
    true_positive = int(np.count_nonzero(labels & predictions))
    false_positive = int(np.count_nonzero(~labels & predictions))
    false_negative = int(np.count_nonzero(labels & ~predictions))
    denominator = 2 * true_positive + false_positive + false_negative
    return 1.0 if denominator == 0 else 2 * true_positive / denominator


def _binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.flatnonzero(labels)
    negatives = np.flatnonzero(~labels)
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    comparisons = scores[positives][:, None] - scores[negatives][None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def summarize_forecasts(rows: list[ForecastMetricRow]) -> dict[str, object]:
    metrics = (
        "ade_m",
        "fde_m",
        "range_mae_m",
        "snr_mae_db",
        "outage_f1",
        "outage_auroc",
        "link_lifetime_error_steps",
    )
    predictors = sorted({row.predictor for row in rows})
    summary: dict[str, object] = {}
    for predictor in predictors:
        selected = [row for row in rows if row.predictor == predictor]
        summary[predictor] = {
            "samples": len(selected),
            **{
                metric: _finite_mean(
                    np.asarray([getattr(row, metric) for row in selected])
                )
                for metric in metrics
            },
        }
    return summary


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def write_forecast_artifacts(
    rows: list[ForecastMetricRow], output_dir: str | Path
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No forecast-evaluation rows were produced.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    json_path = destination / "forecast_metrics.json"
    json_path.write_text(json.dumps(row_dicts, indent=2), encoding="utf-8")
    csv_path = destination / "forecast_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row_dicts[0])
        writer.writeheader()
        writer.writerows(row_dicts)
    summary_path = destination / "forecast_summary.json"
    summary_path.write_text(
        json.dumps(summarize_forecasts(rows), indent=2), encoding="utf-8"
    )
    return {"json": json_path, "csv": csv_path, "summary": summary_path}
