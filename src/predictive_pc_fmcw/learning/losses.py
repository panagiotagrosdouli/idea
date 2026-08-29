from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..link import LinkModel


@dataclass(frozen=True)
class CommunicationLossBreakdown:
    total: float
    trajectory: float
    link: float
    outage: float


def _huber(error: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(error)
    return np.where(
        absolute <= delta,
        0.5 * error**2,
        delta * (absolute - 0.5 * delta),
    )


def communication_aware_loss(
    predicted_relative_xy: ArrayLike,
    target_relative_xy: ArrayLike,
    link_model: LinkModel,
    lambda_link: float = 0.2,
    lambda_outage: float = 0.1,
    huber_delta_m: float = 1.0,
    outage_temperature: float = 0.5,
    ego_heading_rad: ArrayLike | float = 0.0,
) -> CommunicationLossBreakdown:
    predicted = np.asarray(predicted_relative_xy, dtype=np.float64)
    target = np.asarray(target_relative_xy, dtype=np.float64)
    if predicted.shape != target.shape or predicted.shape[-1] != 2:
        raise ValueError("Predicted and target relative trajectories must align.")
    displacement = np.linalg.norm(predicted - target, axis=-1)
    trajectory_loss = float(_huber(displacement, huber_delta_m).mean())

    def link_values(relative: np.ndarray) -> dict[str, np.ndarray]:
        distance = np.linalg.norm(relative, axis=-1)
        bearing = (
            np.arctan2(relative[..., 1], relative[..., 0])
            - np.asarray(ego_heading_rad, dtype=np.float64)
        )
        return link_model.evaluate_arrays(distance, bearing)

    predicted_link = link_values(predicted)
    target_link = link_values(target)
    log_predicted = np.log(np.maximum(predicted_link["snr_linear"], 1e-12))
    log_target = np.log(np.maximum(target_link["snr_linear"], 1e-12))
    link_loss = float(_huber(log_predicted - log_target, 1.0).mean())
    threshold_gamma = -np.log(2 * link_model.config.outage_ber_threshold)
    threshold_log = np.log(max(threshold_gamma, 1e-12))
    logits = (threshold_log - log_predicted) / outage_temperature
    probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    labels = target_link["outage"].astype(np.float64)
    outage_loss = float(
        -np.mean(
            labels * np.log(np.maximum(probability, 1e-12))
            + (1 - labels) * np.log(np.maximum(1 - probability, 1e-12))
        )
    )
    total = trajectory_loss + lambda_link * link_loss + lambda_outage * outage_loss
    return CommunicationLossBreakdown(total, trajectory_loss, link_loss, outage_loss)
