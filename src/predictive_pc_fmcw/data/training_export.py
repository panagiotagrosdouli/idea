from __future__ import annotations

from pathlib import Path

import numpy as np

from ..geometry import heading_from_positions
from .manifest import deterministic_development_split
from .scenario import MotionScenario
from .womd_export import load_womd_motion_scenarios


def build_relative_motion_training_npz(
    womd_json: str | Path,
    output_path: str | Path,
    max_vehicles: int | None = None,
) -> Path:
    scenarios = load_womd_motion_scenarios(womd_json, max_vehicles=max_vehicles)
    return build_training_npz_from_scenarios(
        scenarios,
        output_path,
        source="real_WOMD_motion_proxy_ego_geometry",
    )


def build_training_npz_from_scenarios(
    scenarios: list[MotionScenario],
    output_path: str | Path,
    *,
    source: str,
    fixed_split: str | None = None,
) -> Path:
    histories: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    scenario_ids: list[str] = []
    actor_ids: list[str] = []
    future_ego_headings: list[np.ndarray] = []
    for scenario in scenarios:
        split = scenario.start_index
        ego_heading = heading_from_positions(scenario.ego_positions_xy)
        for vehicle, actor_id in enumerate(scenario.actor_ids):
            relative = (
                scenario.vehicle_positions_xy[:, vehicle]
                - scenario.ego_positions_xy
            )
            histories.append(relative[:split])
            futures.append(relative[split:])
            scenario_ids.append(scenario.scenario_id)
            actor_ids.append(actor_id)
            future_ego_headings.append(ego_heading[split:])
    if not histories:
        raise ValueError("No training samples were produced.")
    unique_scenarios = sorted(set(scenario_ids))
    if fixed_split is None:
        split_by_scenario = {
            scenario_id: deterministic_development_split(scenario_id)
            for scenario_id in unique_scenarios
        }
        if len(unique_scenarios) > 1:
            if "development" not in split_by_scenario.values():
                split_by_scenario[unique_scenarios[-1]] = "development"
            if "training" not in split_by_scenario.values():
                split_by_scenario[unique_scenarios[0]] = "training"
        splits = [split_by_scenario[scenario_id] for scenario_id in scenario_ids]
    else:
        if not fixed_split.strip():
            raise ValueError("fixed_split must be a non-empty label.")
        splits = [fixed_split] * len(scenario_ids)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        history_xy=np.stack(histories).astype(np.float32),
        future_xy=np.stack(futures).astype(np.float32),
        scenario_id=np.asarray(scenario_ids),
        actor_id=np.asarray(actor_ids),
        future_ego_heading_rad=np.stack(future_ego_headings).astype(np.float32),
        split=np.asarray(splits),
        source=np.asarray(source),
        coordinate_frame=np.asarray("world_xy_with_explicit_ego_heading"),
    )
    return destination
