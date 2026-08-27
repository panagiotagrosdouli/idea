from __future__ import annotations

from pathlib import Path

import numpy as np

from .womd_export import load_womd_motion_scenarios


def build_relative_motion_training_npz(
    womd_json: str | Path,
    output_path: str | Path,
    max_vehicles: int | None = None,
) -> Path:
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    scenario_ids: list[str] = []
    actor_ids: list[str] = []
    for scenario in load_womd_motion_scenarios(womd_json, max_vehicles=max_vehicles):
        split = scenario.start_index
        for vehicle, actor_id in enumerate(scenario.actor_ids):
            relative = (
                scenario.vehicle_positions_xy[:, vehicle]
                - scenario.ego_positions_xy
            )
            histories.append(relative[:split])
            futures.append(relative[split:])
            scenario_ids.append(scenario.scenario_id)
            actor_ids.append(actor_id)
    if not histories:
        raise ValueError("No training samples were produced.")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        history_xy=np.stack(histories),
        future_xy=np.stack(futures),
        scenario_id=np.asarray(scenario_ids),
        actor_id=np.asarray(actor_ids),
        source=np.asarray("real_WOMD_motion_proxy_ego_geometry"),
    )
    return destination

