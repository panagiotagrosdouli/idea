from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .scenario import MotionScenario


def iter_tfrecord_payloads(path: str | Path) -> Iterator[bytes]:
    """Read TFRecord payloads without requiring the TensorFlow runtime."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"WOMD TFRecord not found: {source}")
    with source.open("rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise RuntimeError("Invalid TFRecord length header.")
            record_length = struct.unpack("<Q", length_bytes)[0]
            if len(handle.read(4)) != 4:
                raise RuntimeError("Missing TFRecord length CRC.")
            payload = handle.read(record_length)
            if len(payload) != record_length:
                raise RuntimeError("Incomplete TFRecord payload.")
            if len(handle.read(4)) != 4:
                raise RuntimeError("Missing TFRecord data CRC.")
            yield payload


def iter_official_womd_protos(
    paths: Sequence[str | Path], max_scenarios: int | None = None
) -> Iterator[Any]:
    """Yield official WOMD Scenario protos from one or more TFRecord shards."""

    try:
        from waymo_open_dataset.protos import scenario_pb2

        scenario_class = scenario_pb2.Scenario
    except ImportError:  # Python 3.13/Colab has no compatible Waymo TF wheel.
        from .womd_minimal_proto import scenario_message_class

        scenario_class = scenario_message_class()
    produced = 0
    for path in paths:
        for payload in iter_tfrecord_payloads(path):
            scenario = scenario_class()
            scenario.ParseFromString(payload)
            yield scenario
            produced += 1
            if max_scenarios is not None and produced >= max_scenarios:
                return


def scenario_proto_to_motion_scenario(
    scenario: Any,
    max_vehicles: int | None = None,
    require_vehicle_type: bool = True,
) -> MotionScenario | None:
    """Convert an official proto using the true SDC and strict valid masks.

    Tracks with any invalid state in the retained window are excluded instead
    of interpolated. This conservative first adapter keeps validity explicit in
    its selection rule; future work can add masked sequence models.
    """

    timestamps = np.asarray(scenario.timestamps_seconds, dtype=np.float64)
    total = timestamps.size
    start_index = int(scenario.current_time_index) + 1
    if total < 4 or not 2 <= start_index < total:
        return None
    sdc_index = int(scenario.sdc_track_index)
    if not 0 <= sdc_index < len(scenario.tracks):
        raise ValueError("Official WOMD scenario has an invalid sdc_track_index.")

    def valid_xy(track: Any) -> np.ndarray | None:
        if len(track.states) < total:
            return None
        states = track.states[:total]
        if not all(bool(state.valid) for state in states):
            return None
        values = np.asarray(
            [[float(state.center_x), float(state.center_y)] for state in states],
            dtype=np.float64,
        )
        return values if np.all(np.isfinite(values)) else None

    ego = valid_xy(scenario.tracks[sdc_index])
    if ego is None:
        return None
    anchor = start_index - 1
    candidates: list[tuple[float, int, np.ndarray]] = []
    for index, track in enumerate(scenario.tracks):
        if index == sdc_index:
            continue
        if require_vehicle_type and int(track.object_type) != 1:
            continue
        positions = valid_xy(track)
        if positions is None:
            continue
        distance = float(np.linalg.norm(positions[anchor] - ego[anchor]))
        candidates.append((distance, index, positions))
    candidates.sort(key=lambda item: (item[0], item[1]))
    if max_vehicles is not None:
        candidates = candidates[:max_vehicles]
    if not candidates:
        return None
    return MotionScenario(
        scenario_id=str(scenario.scenario_id),
        timestamps_s=timestamps,
        ego_positions_xy=ego,
        vehicle_positions_xy=np.stack(
            [positions for _, _, positions in candidates], axis=1
        ),
        actor_ids=tuple(
            str(scenario.tracks[index].id) for _, index, _ in candidates
        ),
        start_index=start_index,
        source="real_WOMD_v1.3.1_true_SDC_model_based_link",
    )


def load_official_womd_tfrecords(
    paths: Sequence[str | Path],
    *,
    max_scenarios: int | None = None,
    max_vehicles: int | None = None,
) -> list[MotionScenario]:
    scenarios: list[MotionScenario] = []
    for proto in iter_official_womd_protos(paths, max_scenarios=max_scenarios):
        converted = scenario_proto_to_motion_scenario(
            proto, max_vehicles=max_vehicles
        )
        if converted is not None:
            scenarios.append(converted)
    return scenarios
