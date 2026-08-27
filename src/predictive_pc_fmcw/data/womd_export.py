from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .scenario import MotionScenario


def _medoid_index(points: np.ndarray) -> int:
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    return int(np.argmin(pairwise.sum(axis=1)))


def load_womd_motion_scenarios(
    path: str | Path,
    max_vehicles: int | None = None,
    dt_s: float = 0.1,
) -> list[MotionScenario]:
    """Load the compact Stage-5 real-WOMD motion export.

    The supplied export does not retain the SDC/ego identifier. For downstream
    software validation only, the current-position medoid is selected as a
    deterministic proxy ego. Publications must label results from this adapter
    as real motion with proxy geometry, not as measured optical communication.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        records: list[dict[str, Any]] = json.load(handle)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        required = {"scenario_id", "track_index", "past", "future"}
        if not required.issubset(record):
            raise ValueError(f"Malformed WOMD export record: {required - set(record)}")
        grouped[str(record["scenario_id"])].append(record)

    scenarios: list[MotionScenario] = []
    for scenario_id, actors in sorted(grouped.items()):
        trajectories = [
            np.asarray(actor["past"] + actor["future"], dtype=np.float64)
            for actor in actors
        ]
        lengths = {trajectory.shape for trajectory in trajectories}
        if len(lengths) != 1 or next(iter(lengths))[1:] != (2,):
            raise ValueError(f"Inconsistent trajectories in scenario {scenario_id}.")
        history_steps = len(actors[0]["past"])
        current = np.stack(
            [trajectory[history_steps - 1] for trajectory in trajectories]
        )
        ego_index = _medoid_index(current)
        candidate_indices = [
            index for index in range(len(actors)) if index != ego_index
        ]
        candidate_indices.sort(
            key=lambda index: float(np.linalg.norm(current[index] - current[ego_index]))
        )
        if max_vehicles is not None:
            candidate_indices = candidate_indices[:max_vehicles]
        if not candidate_indices:
            continue
        total = trajectories[0].shape[0]
        scenarios.append(
            MotionScenario(
                scenario_id=scenario_id,
                timestamps_s=np.arange(total, dtype=np.float64) * dt_s,
                ego_positions_xy=trajectories[ego_index],
                vehicle_positions_xy=np.stack(
                    [trajectories[index] for index in candidate_indices], axis=1
                ),
                actor_ids=tuple(
                    str(actors[index]["track_index"]) for index in candidate_indices
                ),
                start_index=history_steps,
                source="real_WOMD_motion_proxy_ego_geometry",
            )
        )
    return scenarios
