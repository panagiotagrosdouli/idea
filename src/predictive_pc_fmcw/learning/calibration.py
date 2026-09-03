from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualGaussianCalibration:
    """Zero-mean diagonal residual Gaussian fitted on development data only."""

    variance_xy: np.ndarray
    minimum_variance: float = 1e-6

    def to_dict(self) -> dict[str, object]:
        return {
            "method": "per-horizon zero-mean diagonal residual Gaussian",
            "fit_split": "development",
            "variance_xy": self.variance_xy.tolist(),
            "minimum_variance": self.minimum_variance,
        }


def fit_residual_gaussian(
    predictions_xy: np.ndarray,
    targets_xy: np.ndarray,
    *,
    minimum_variance: float = 1e-6,
) -> ResidualGaussianCalibration:
    predicted = np.asarray(predictions_xy, dtype=np.float64)
    target = np.asarray(targets_xy, dtype=np.float64)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 3
        or predicted.shape[-1] != 2
    ):
        raise ValueError(
            "Predictions and targets must have shape (samples, horizon, 2)."
        )
    if predicted.shape[0] < 2 or minimum_variance <= 0:
        raise ValueError(
            "At least two samples and positive minimum_variance are required."
        )
    residual = target - predicted
    variance = np.mean(np.square(residual), axis=0)
    variance = np.maximum(variance, minimum_variance)
    if not np.all(np.isfinite(variance)):
        raise ValueError("Residual calibration produced non-finite variance.")
    return ResidualGaussianCalibration(variance, minimum_variance)


def gaussian_nll_and_coverage(
    predictions_xy: np.ndarray,
    targets_xy: np.ndarray,
    calibration: ResidualGaussianCalibration,
) -> dict[str, np.ndarray]:
    predicted = np.asarray(predictions_xy, dtype=np.float64)
    target = np.asarray(targets_xy, dtype=np.float64)
    variance = np.asarray(calibration.variance_xy, dtype=np.float64)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 3
        or predicted.shape[-1] != 2
    ):
        raise ValueError(
            "Predictions and targets must have shape (samples, horizon, 2)."
        )
    if variance.shape != predicted.shape[1:] or np.any(variance <= 0):
        raise ValueError(
            "Calibration variance must be positive with shape (horizon, 2)."
        )
    squared_mahalanobis = np.sum(np.square(target - predicted) / variance, axis=-1)
    point_nll = 0.5 * (
        2.0 * np.log(2.0 * np.pi)
        + np.sum(np.log(variance), axis=-1)[None, :]
        + squared_mahalanobis
    )
    # Chi-square(2) quantiles for joint x/y ellipses.
    thresholds = {50: 1.38629436112, 90: 4.60517018599, 95: 5.99146454711}
    return {
        "nll": point_nll,
        **{
            f"coverage_{level}": squared_mahalanobis <= threshold
            for level, threshold in thresholds.items()
        },
    }
