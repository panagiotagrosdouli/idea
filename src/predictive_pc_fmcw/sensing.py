from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SensingConfig
from .geometry import heading_from_positions, range_and_bearing


@dataclass(frozen=True)
class ObservationBatch:
    """Observed positions and an isotropic-equivalent position uncertainty."""

    positions_xy: NDArray[np.float64]
    position_std_m: NDArray[np.float64]
    model: str
    measured_data: bool = False


def _ar1_standard_normal(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    correlation: float,
) -> NDArray[np.float64]:
    innovations = rng.normal(size=shape)
    if correlation == 0 or shape[0] < 2:
        return innovations
    scale = float(np.sqrt(1.0 - correlation**2))
    for index in range(1, shape[0]):
        innovations[index] = (
            correlation * innovations[index - 1]
            + scale * innovations[index]
        )
    return innovations


def observe_combined_history(
    combined_positions_xy: NDArray[np.float64],
    config: SensingConfig,
    seed: int,
) -> ObservationBatch:
    """Apply a declared synthetic observation model to target histories.

    The ego pose is kept exact.  For the range/bearing model, target noise is
    sampled in the instantaneous ego frame and transformed back to world
    coordinates.  The returned scalar standard deviation is the square root
    of half the positional covariance trace and can parameterize an isotropic
    Kalman measurement covariance without pretending that it is exact.
    """

    positions = np.asarray(combined_positions_xy, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 2 or positions.shape[1] < 2:
        raise ValueError("combined_positions_xy must have shape (time, actors, 2).")
    observed = positions.copy()
    time_steps, actors, _ = positions.shape
    target_count = actors - 1
    equivalent_std = np.zeros((time_steps, actors), dtype=np.float64)
    if config.model == "perfect":
        return ObservationBatch(observed, equivalent_std, config.model)

    rng = np.random.default_rng(seed)
    correlation = config.temporal_correlation
    if config.model == "cartesian_iid":
        noise = _ar1_standard_normal(
            rng, (time_steps, target_count, 2), correlation
        ) * config.cartesian_std_m
        observed[:, 1:] += noise
        equivalent_std[:, 1:] = config.cartesian_std_m
        return ObservationBatch(observed, equivalent_std, config.model)

    headings = heading_from_positions(positions[:, 0])
    distances, bearings = range_and_bearing(
        positions[:, 1:], positions[:, :1], headings[:, None]
    )
    range_std = config.range_std_base_m + config.range_std_per_m * distances
    bearing_std_rad = np.deg2rad(config.bearing_std_deg)
    range_error = _ar1_standard_normal(
        rng, (time_steps, target_count), correlation
    ) * range_std
    bearing_error = _ar1_standard_normal(
        rng, (time_steps, target_count), correlation
    ) * bearing_std_rad
    noisy_range = np.maximum(distances + range_error, 0.0)
    noisy_world_bearing = bearings + bearing_error + headings[:, None]
    relative = np.stack(
        [
            noisy_range * np.cos(noisy_world_bearing),
            noisy_range * np.sin(noisy_world_bearing),
        ],
        axis=-1,
    )
    observed[:, 1:] = positions[:, :1] + relative
    tangential_std = distances * bearing_std_rad
    equivalent_std[:, 1:] = np.sqrt(
        0.5 * (range_std**2 + tangential_std**2)
    )
    return ObservationBatch(observed, equivalent_std, config.model)
