from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..scheduling.base import SchedulerContext


@dataclass(frozen=True)
class ObservationConfig:
    """Normalization constants for the causal RL scheduler observation."""

    max_queue_packets: int
    max_deadline_steps: int
    prediction_horizon_steps: int
    include_prediction: bool = True

    def __post_init__(self) -> None:
        if self.max_queue_packets <= 0:
            raise ValueError("max_queue_packets must be positive")
        if self.max_deadline_steps <= 0:
            raise ValueError("max_deadline_steps must be positive")
        if self.prediction_horizon_steps <= 0:
            raise ValueError("prediction_horizon_steps must be positive")


def feature_names(include_prediction: bool = True) -> tuple[str, ...]:
    names = (
        "eligible",
        "queue_fraction",
        "deadline_urgency",
        "current_goodput_fraction",
        "current_outage",
        "delivered_share",
        "previous_vehicle",
    )
    if include_prediction:
        names += (
            "predicted_mean_goodput_fraction",
            "predicted_outage_fraction",
            "predicted_lifetime_fraction",
        )
    return names


def _safe_fraction(values: NDArray[np.float64], denominator: float) -> NDArray[np.float64]:
    return np.clip(values / max(float(denominator), 1e-12), 0.0, 1.0)


def build_observation(
    context: SchedulerContext,
    config: ObservationConfig,
) -> NDArray[np.float32]:
    """Build a per-vehicle, causal observation matrix.

    Rows correspond to vehicles.  No ground-truth future link state is used:
    predictive features come only from ``SchedulerContext.predicted_*``.
    """

    vehicles = context.vehicles
    eligible = (context.queue_lengths > 0).astype(np.float64)
    queue_fraction = _safe_fraction(
        context.queue_lengths.astype(np.float64), config.max_queue_packets
    )

    deadline = context.time_to_deadline.astype(np.float64)
    deadline = np.where(np.isfinite(deadline), deadline, config.max_deadline_steps)
    deadline_fraction = _safe_fraction(deadline, config.max_deadline_steps)
    deadline_urgency = eligible * (1.0 - deadline_fraction)

    current_goodput = _safe_fraction(
        context.current_goodput_bps.astype(np.float64), context.data_rate_bps
    )
    current_outage = context.current_outage.astype(np.float64)

    total_delivered = float(np.sum(context.delivered_bits))
    if total_delivered > 0.0:
        delivered_share = context.delivered_bits.astype(np.float64) / total_delivered
    else:
        delivered_share = np.zeros(vehicles, dtype=np.float64)

    previous_vehicle = np.zeros(vehicles, dtype=np.float64)
    if context.previous_vehicle is not None:
        if not 0 <= context.previous_vehicle < vehicles:
            raise ValueError("previous_vehicle is outside the vehicle range")
        previous_vehicle[context.previous_vehicle] = 1.0

    columns: list[NDArray[np.float64]] = [
        eligible,
        queue_fraction,
        deadline_urgency,
        current_goodput,
        current_outage,
        delivered_share,
        previous_vehicle,
    ]

    if config.include_prediction:
        predicted_goodput = np.asarray(context.predicted_goodput_bps, dtype=np.float64)
        predicted_outage = np.asarray(context.predicted_outage, dtype=np.float64)
        if predicted_goodput.ndim != 2 or predicted_goodput.shape[0] != vehicles:
            raise ValueError("predicted_goodput_bps must have shape (vehicles, horizon)")
        if predicted_outage.shape != predicted_goodput.shape:
            raise ValueError("predicted_outage must match predicted_goodput_bps")

        predicted_mean_goodput = _safe_fraction(
            np.mean(predicted_goodput, axis=1), context.data_rate_bps
        )
        predicted_outage_fraction = np.mean(predicted_outage, axis=1)
        predicted_lifetime_fraction = _safe_fraction(
            context.predicted_lifetime_steps.astype(np.float64),
            config.prediction_horizon_steps,
        )
        columns.extend(
            [
                predicted_mean_goodput,
                predicted_outage_fraction,
                predicted_lifetime_fraction,
            ]
        )

    observation = np.column_stack(columns).astype(np.float32, copy=False)
    expected = (vehicles, len(feature_names(config.include_prediction)))
    if observation.shape != expected:
        raise AssertionError(f"Unexpected observation shape {observation.shape}, expected {expected}")
    if not np.all(np.isfinite(observation)):
        raise ValueError("RL observation contains non-finite values")
    return observation
