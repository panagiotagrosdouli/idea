from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TrajectoryPredictor(Protocol):
    name: str

    def predict(
        self, history_xy: ArrayLike, horizon_steps: int, dt_s: float
    ) -> NDArray[np.float64]: ...


def _validate_history(history_xy: ArrayLike) -> NDArray[np.float64]:
    history = np.asarray(history_xy, dtype=np.float64)
    if history.ndim < 2 or history.shape[-1] != 2 or history.shape[-2] < 2:
        raise ValueError("history_xy must have shape (..., history, 2), history >= 2.")
    return history


@dataclass(frozen=True)
class ConstantVelocityPredictor:
    name: str = "constant_velocity"

    def predict(
        self, history_xy: ArrayLike, horizon_steps: int, dt_s: float
    ) -> NDArray[np.float64]:
        history = _validate_history(history_xy)
        velocity = (history[..., -1, :] - history[..., -2, :]) / dt_s
        steps = np.arange(1, horizon_steps + 1, dtype=np.float64)
        return history[..., -1, None, :] + velocity[..., None, :] * (
            steps * dt_s
        )[..., None]


@dataclass(frozen=True)
class ConstantAccelerationPredictor:
    acceleration_clip_mps2: float = 6.0
    name: str = "constant_acceleration"

    def predict(
        self, history_xy: ArrayLike, horizon_steps: int, dt_s: float
    ) -> NDArray[np.float64]:
        history = _validate_history(history_xy)
        if history.shape[-2] < 3:
            return ConstantVelocityPredictor().predict(history, horizon_steps, dt_s)
        velocity_now = (history[..., -1, :] - history[..., -2, :]) / dt_s
        velocity_before = (history[..., -2, :] - history[..., -3, :]) / dt_s
        acceleration = (velocity_now - velocity_before) / dt_s
        norm = np.linalg.norm(acceleration, axis=-1, keepdims=True)
        acceleration = acceleration * np.minimum(
            1.0, self.acceleration_clip_mps2 / np.maximum(norm, 1e-12)
        )
        tau = np.arange(1, horizon_steps + 1, dtype=np.float64) * dt_s
        return (
            history[..., -1, None, :]
            + velocity_now[..., None, :] * tau[..., None]
            + 0.5 * acceleration[..., None, :] * tau[..., None] ** 2
        )


@dataclass(frozen=True)
class ForecastBundle:
    ego_xy: NDArray[np.float64]
    vehicle_xy: NDArray[np.float64]
    future_indices: NDArray[np.int64]
    oracle: bool


def forecast_scenario(
    combined_positions: NDArray[np.float64],
    time_index: int,
    horizon_steps: int,
    dt_s: float,
    predictor: TrajectoryPredictor | None,
    oracle: bool = False,
) -> ForecastBundle:
    total = combined_positions.shape[0]
    indices = np.minimum(
        np.arange(time_index + 1, time_index + horizon_steps + 1), total - 1
    ).astype(np.int64)
    if oracle:
        prediction = combined_positions[indices].transpose(1, 0, 2)
    else:
        if predictor is None:
            current = combined_positions[time_index].transpose(0, 1)
            prediction = np.repeat(current[:, None, :], horizon_steps, axis=1)
        else:
            causal_history = combined_positions[: time_index + 1].transpose(1, 0, 2)
            prediction = predictor.predict(causal_history, horizon_steps, dt_s)
    return ForecastBundle(
        ego_xy=prediction[0],
        vehicle_xy=prediction[1:],
        future_indices=indices,
        oracle=oracle,
    )

