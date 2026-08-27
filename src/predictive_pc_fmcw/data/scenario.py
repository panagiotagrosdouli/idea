from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class MotionScenario:
    scenario_id: str
    timestamps_s: NDArray[np.float64]
    ego_positions_xy: NDArray[np.float64]
    vehicle_positions_xy: NDArray[np.float64]
    actor_ids: tuple[str, ...]
    start_index: int
    source: str = "synthetic"

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        ego = np.asarray(self.ego_positions_xy, dtype=np.float64)
        vehicles = np.asarray(self.vehicle_positions_xy, dtype=np.float64)
        if timestamps.ndim != 1:
            raise ValueError("timestamps_s must be one-dimensional.")
        if ego.shape != (timestamps.size, 2):
            raise ValueError("ego_positions_xy must have shape (time, 2).")
        if vehicles.ndim != 3 or vehicles.shape[0] != timestamps.size:
            raise ValueError(
                "vehicle_positions_xy must have shape (time, vehicles, 2)."
            )
        if vehicles.shape[2] != 2 or vehicles.shape[1] != len(self.actor_ids):
            raise ValueError("Vehicle positions and actor identifiers are inconsistent.")
        if not 2 <= self.start_index < timestamps.size:
            raise ValueError("start_index must leave history and evaluation samples.")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing.")

    @property
    def vehicle_count(self) -> int:
        return self.vehicle_positions_xy.shape[1]

    @property
    def evaluation_slots(self) -> int:
        return self.timestamps_s.size - self.start_index

    @property
    def dt_s(self) -> float:
        return float(np.median(np.diff(self.timestamps_s)))

    def combined_positions(self) -> NDArray[np.float64]:
        return np.concatenate(
            [self.ego_positions_xy[:, None, :], self.vehicle_positions_xy], axis=1
        )

