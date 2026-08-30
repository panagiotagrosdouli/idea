from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data.scenario import MotionScenario
from .predictors import TrajectoryPredictor, forecast_scenario

CHI2_DF2_THRESHOLDS = {
    "coverage_50": 1.3862943611198906,
    "coverage_90": 4.605170185988092,
    "coverage_95": 5.991464547107979,
}


@dataclass(frozen=True)
class GaussianCalibration:
    predictor: str
    horizon_steps: int
    variance_floor_m2: float
    isotropic_variance_m2: tuple[float, ...]
    calibration_scenario_ids: tuple[str, ...]


@dataclass(frozen=True)
class ProbabilisticMetricRow:
    predictor: str
    calibration_scenarios: int
    evaluation_scenarios: int
    calibration_evaluation_overlap: int
    samples: int
    rmse_m: float
    mean_nll: float
    mean_sigma_m: float
    coverage_50: float
    coverage_90: float
    coverage_95: float
    calibration_error: float


def fit_isotropic_gaussian_calibration(
    scenarios: Iterable[MotionScenario],
    predictor: TrajectoryPredictor,
    horizon_steps: int,
    *,
    anchor_stride: int = 5,
    variance_floor_m2: float = 1e-4,
) -> GaussianCalibration:
    """Fit per-horizon isotropic residual variance on calibration scenarios."""

    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one calibration scenario is required.")
    if horizon_steps < 1 or anchor_stride < 1 or variance_floor_m2 <= 0:
        raise ValueError("Horizon, stride, and variance floor must be positive.")
    squared_errors = [[] for _ in range(horizon_steps)]
    for scenario in scenario_list:
        for step, residual in _relative_residuals(
            scenario, predictor, horizon_steps, anchor_stride
        ):
            squared_errors[step].extend(np.sum(residual**2, axis=-1).tolist())
    variances = []
    previous = variance_floor_m2
    for values in squared_errors:
        estimate = (
            float(np.mean(values)) / 2.0 if values else previous
        )
        # A non-decreasing envelope avoids claiming increased certainty farther
        # into an open-loop rollout because of finite calibration support.
        previous = max(previous, estimate, variance_floor_m2)
        variances.append(previous)
    return GaussianCalibration(
        predictor=predictor.name,
        horizon_steps=horizon_steps,
        variance_floor_m2=variance_floor_m2,
        isotropic_variance_m2=tuple(variances),
        calibration_scenario_ids=tuple(
            sorted(scenario.scenario_id for scenario in scenario_list)
        ),
    )


def evaluate_gaussian_calibration(
    scenarios: Iterable[MotionScenario],
    predictor: TrajectoryPredictor,
    calibration: GaussianCalibration,
    *,
    anchor_stride: int = 5,
) -> ProbabilisticMetricRow:
    """Evaluate 2-D Gaussian NLL and coverage on disjoint scenarios."""

    scenario_list = list(scenarios)
    if not scenario_list:
        raise ValueError("At least one evaluation scenario is required.")
    if predictor.name != calibration.predictor:
        raise ValueError("Predictor and calibration names do not match.")
    if anchor_stride < 1:
        raise ValueError("anchor_stride must be positive.")
    evaluation_ids = {scenario.scenario_id for scenario in scenario_list}
    overlap = evaluation_ids & set(calibration.calibration_scenario_ids)
    squared_errors: list[float] = []
    mahalanobis_squared: list[float] = []
    nll_values: list[float] = []
    sigma_values: list[float] = []
    for scenario in scenario_list:
        for step, residual in _relative_residuals(
            scenario,
            predictor,
            calibration.horizon_steps,
            anchor_stride,
        ):
            variance = calibration.isotropic_variance_m2[step]
            squared = np.sum(residual**2, axis=-1)
            squared_errors.extend(squared.tolist())
            mahalanobis_squared.extend((squared / variance).tolist())
            nll_values.extend(
                (squared / (2.0 * variance) + np.log(2.0 * np.pi * variance)).tolist()
            )
            sigma_values.extend([float(np.sqrt(variance))] * residual.shape[0])
    if not squared_errors:
        raise ValueError("No evaluable forecast residuals were produced.")
    mahalanobis = np.asarray(mahalanobis_squared, dtype=np.float64)
    coverage = {
        name: float(np.mean(mahalanobis <= threshold))
        for name, threshold in CHI2_DF2_THRESHOLDS.items()
    }
    nominal = {"coverage_50": 0.5, "coverage_90": 0.9, "coverage_95": 0.95}
    return ProbabilisticMetricRow(
        predictor=predictor.name,
        calibration_scenarios=len(calibration.calibration_scenario_ids),
        evaluation_scenarios=len(evaluation_ids),
        calibration_evaluation_overlap=len(overlap),
        samples=len(squared_errors),
        rmse_m=float(np.sqrt(np.mean(squared_errors))),
        mean_nll=float(np.mean(nll_values)),
        mean_sigma_m=float(np.mean(sigma_values)),
        coverage_50=coverage["coverage_50"],
        coverage_90=coverage["coverage_90"],
        coverage_95=coverage["coverage_95"],
        calibration_error=float(
            np.mean([abs(coverage[key] - value) for key, value in nominal.items()])
        ),
    )


def evaluate_probabilistic_baselines(
    calibration_scenarios: Iterable[MotionScenario],
    evaluation_scenarios: Iterable[MotionScenario],
    predictors: Mapping[str, TrajectoryPredictor],
    horizon_steps: int,
    *,
    anchor_stride: int = 5,
) -> tuple[list[GaussianCalibration], list[ProbabilisticMetricRow]]:
    calibration_list = list(calibration_scenarios)
    evaluation_list = list(evaluation_scenarios)
    calibrations = []
    rows = []
    for predictor in predictors.values():
        fitted = fit_isotropic_gaussian_calibration(
            calibration_list,
            predictor,
            horizon_steps,
            anchor_stride=anchor_stride,
        )
        calibrations.append(fitted)
        rows.append(
            evaluate_gaussian_calibration(
                evaluation_list,
                predictor,
                fitted,
                anchor_stride=anchor_stride,
            )
        )
    return calibrations, rows


def write_probabilistic_artifacts(
    calibrations: list[GaussianCalibration],
    rows: list[ProbabilisticMetricRow],
    output_dir: str | Path,
) -> dict[str, Path]:
    if not calibrations or not rows:
        raise ValueError("Probabilistic artifacts require calibration and rows.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scope": "scenario-safe classical Gaussian residual calibration",
        "learned_gaussian_or_gmm_checkpoint_used": False,
        "calibrations": [asdict(item) for item in calibrations],
        "metrics": [asdict(item) for item in rows],
    }
    json_path = destination / "probabilistic_calibration.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = destination / "probabilistic_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return {"json": json_path, "csv": csv_path}


def _relative_residuals(
    scenario: MotionScenario,
    predictor: TrajectoryPredictor,
    horizon_steps: int,
    anchor_stride: int,
):
    combined = scenario.combined_positions()
    first_anchor = max(2, scenario.start_index - 1)
    for anchor in range(first_anchor, combined.shape[0] - 1, anchor_stride):
        horizon = min(horizon_steps, combined.shape[0] - anchor - 1)
        bundle = forecast_scenario(
            combined,
            anchor,
            horizon,
            scenario.dt_s,
            predictor,
        )
        predicted = np.concatenate(
            [bundle.ego_xy[None, :, :], bundle.vehicle_xy], axis=0
        )
        indices = np.arange(anchor + 1, anchor + horizon + 1)
        actual = combined[indices].transpose(1, 0, 2)
        residual = (
            predicted[1:] - predicted[:1] - (actual[1:] - actual[:1])
        )
        for step in range(horizon):
            yield step, residual[:, step, :]
