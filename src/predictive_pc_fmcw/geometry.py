from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def wrap_angle_rad(angle: ArrayLike) -> NDArray[np.float64]:
    value = np.asarray(angle, dtype=np.float64)
    return (value + np.pi) % (2 * np.pi) - np.pi


def range_and_bearing(
    target_xy: ArrayLike,
    ego_xy: ArrayLike,
    ego_heading_rad: ArrayLike | float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    target = np.asarray(target_xy, dtype=np.float64)
    ego = np.asarray(ego_xy, dtype=np.float64)
    if target.shape[-1] != 2 or ego.shape[-1] != 2:
        raise ValueError("Coordinates must have final dimension 2.")
    delta = target - ego
    distance = np.linalg.norm(delta, axis=-1)
    bearing = wrap_angle_rad(
        np.arctan2(delta[..., 1], delta[..., 0])
        - np.asarray(ego_heading_rad, dtype=np.float64)
    )
    return distance, bearing


def heading_from_positions(xy: ArrayLike, fallback: float = 0.0) -> NDArray[np.float64]:
    positions = np.asarray(xy, dtype=np.float64)
    if positions.ndim < 2 or positions.shape[-1] != 2:
        raise ValueError("xy must have shape (..., time, 2).")
    delta = np.diff(positions, axis=-2, prepend=positions[..., :1, :])
    speed = np.linalg.norm(delta, axis=-1)
    heading = np.arctan2(delta[..., 1], delta[..., 0])
    heading = np.where(speed > 1e-9, heading, np.nan)
    flat = heading.reshape(-1, heading.shape[-1])
    for row in flat:
        last = fallback
        for index in range(row.size):
            if np.isfinite(row[index]):
                last = float(row[index])
            else:
                row[index] = last
    return wrap_angle_rad(heading)

