from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .data.scenario import MotionScenario
from .geometry import heading_from_positions, range_and_bearing, wrap_angle_rad


@dataclass(frozen=True)
class ScenarioSliceRow:
    scenario_id: str
    actor_id: str
    approaching: bool
    receding: bool
    straight: bool
    lane_change_or_merge: bool
    turn: bool
    fov_edge: bool
    dense: bool
    range_change_m: float
    lateral_change_m: float
    heading_change_deg: float
    labels: tuple[str, ...]


def classify_scenario_slices(
    scenario: MotionScenario,
    *,
    field_of_view_deg: float = 70.0,
    fov_edge_margin_deg: float = 5.0,
    range_change_threshold_m: float = 2.0,
    lateral_change_threshold_m: float = 1.5,
    turn_threshold_deg: float = 15.0,
    dense_vehicle_threshold: int = 8,
) -> list[ScenarioSliceRow]:
    ego_heading = heading_from_positions(scenario.ego_positions_xy)
    distances, bearings = range_and_bearing(
        scenario.vehicle_positions_xy,
        scenario.ego_positions_xy[:, None, :],
        ego_heading[:, None],
    )
    relative = scenario.vehicle_positions_xy - scenario.ego_positions_xy[:, None, :]
    cosine = np.cos(ego_heading)[:, None]
    sine = np.sin(ego_heading)[:, None]
    lateral = -sine * relative[..., 0] + cosine * relative[..., 1]
    half_fov = field_of_view_deg / 2
    rows: list[ScenarioSliceRow] = []
    for vehicle, actor_id in enumerate(scenario.actor_ids):
        range_change = float(distances[-1, vehicle] - distances[0, vehicle])
        lateral_change = float(lateral[-1, vehicle] - lateral[0, vehicle])
        target_heading = heading_from_positions(
            scenario.vehicle_positions_xy[:, vehicle]
        )
        heading_change = float(
            np.rad2deg(
                abs(wrap_angle_rad(target_heading[-1] - target_heading[0]))
            )
        )
        approaching = range_change < -range_change_threshold_m
        receding = range_change > range_change_threshold_m
        lane_change = abs(lateral_change) >= lateral_change_threshold_m
        turn = heading_change >= turn_threshold_deg
        straight = heading_change <= 5.0 and abs(lateral_change) < 0.75
        bearing_deg = np.abs(np.rad2deg(bearings[:, vehicle]))
        fov_edge = bool(
            np.any(np.abs(bearing_deg - half_fov) <= fov_edge_margin_deg)
        )
        dense = scenario.vehicle_count >= dense_vehicle_threshold
        flags = {
            "approaching": approaching,
            "receding": receding,
            "straight": straight,
            "lane_change_or_merge": lane_change,
            "turn": turn,
            "fov_edge": fov_edge,
            "dense": dense,
        }
        labels = tuple(name for name, active in flags.items() if active)
        if not labels:
            labels = ("other",)
        rows.append(
            ScenarioSliceRow(
                scenario_id=scenario.scenario_id,
                actor_id=actor_id,
                approaching=approaching,
                receding=receding,
                straight=straight,
                lane_change_or_merge=lane_change,
                turn=turn,
                fov_edge=fov_edge,
                dense=dense,
                range_change_m=range_change,
                lateral_change_m=lateral_change,
                heading_change_deg=heading_change,
                labels=labels,
            )
        )
    return rows


def write_scenario_slice_artifacts(
    rows: list[ScenarioSliceRow], output_dir: str | Path
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No scenario-slice rows were produced.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    serialized = [
        {**asdict(row), "labels": list(row.labels)} for row in rows
    ]
    json_path = destination / "scenario_slices.json"
    json_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    csv_path = destination / "scenario_slices.csv"
    csv_rows = [
        {**row, "labels": ";".join(row["labels"])} for row in serialized
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    return {"json": json_path, "csv": csv_path}
