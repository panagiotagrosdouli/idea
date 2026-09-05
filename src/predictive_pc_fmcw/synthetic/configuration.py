"""Canonical JSON loading for Synthetic Dataset v1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import FAMILIES, DatasetBuildConfig
from .mobility import SyntheticMobilityConfig
from .observations import ObservationNoiseConfig


@dataclass(frozen=True)
class LoadedSyntheticProtocol:
    build_config: DatasetBuildConfig
    sha256: str


def _pair(raw: object, name: str) -> tuple[float, float]:
    if not isinstance(raw, list) or len(raw) != 2:
        raise ValueError(f"{name} must contain exactly two numeric bounds")
    return float(raw[0]), float(raw[1])


def load_synthetic_protocol_config(path: str | Path) -> LoadedSyntheticProtocol:
    """Load the frozen JSON instead of duplicating mobility/noise defaults."""
    source = Path(path)
    raw_bytes = source.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    if raw.get("protocol") != "synthetic_dataset_v1":
        raise ValueError("synthetic protocol config has an unexpected protocol id")

    simulation = raw["simulation"]
    mobility = raw["mobility"]
    train = mobility["train_dev_ranges"]
    ood = mobility["ood_ranges"]
    families = tuple(mobility["families"])
    if families != FAMILIES:
        raise ValueError("canonical mobility-family order differs from implementation")

    splits = raw["splits"]
    train_fraction = float(splits["train_fraction"])
    development_fraction = float(splits["development_fraction"])
    if not np.isclose(train_fraction, 0.70) or not np.isclose(
        development_fraction, 0.15
    ):
        raise ValueError(
            "current dataset builder freezes split fractions at 0.70/0.15"
        )
    if splits.get("official_test_for_selection") is not False:
        raise ValueError("official held-out test must remain excluded from selection")
    if splits.get("ood_test_for_selection") is not False:
        raise ValueError("OOD test must remain excluded from selection")

    duration_s = float(simulation["duration_s"])
    sampling_hz = float(simulation["sampling_hz"])
    train_mobility = SyntheticMobilityConfig(
        duration_s=duration_s,
        sampling_hz=sampling_hz,
        initial_range_m=_pair(train["initial_range_m"], "initial_range_m"),
        speed_mps=_pair(train["speed_mps"], "speed_mps"),
        acceleration_mps2=_pair(train["acceleration_mps2"], "acceleration_mps2"),
        lateral_speed_mps=_pair(train["lateral_speed_mps"], "lateral_speed_mps"),
        maneuver_duration_s=_pair(
            train["maneuver_duration_s"], "maneuver_duration_s"
        ),
    )
    ood_mobility = SyntheticMobilityConfig(
        duration_s=duration_s,
        sampling_hz=sampling_hz,
        initial_range_m=_pair(ood["initial_range_m"], "ood.initial_range_m"),
        speed_mps=_pair(ood["speed_mps"], "ood.speed_mps"),
        acceleration_mps2=_pair(ood["acceleration_mps2"], "ood.acceleration_mps2"),
        lateral_speed_mps=_pair(
            ood["lateral_speed_mps"], "ood.lateral_speed_mps"
        ),
        maneuver_duration_s=_pair(
            ood["maneuver_duration_s"], "ood.maneuver_duration_s"
        ),
    )
    noise = raw["observation_noise"]
    observations = ObservationNoiseConfig(
        range_std_m=float(noise["range_std_m"]),
        radial_velocity_std_mps=float(noise["radial_velocity_std_mps"]),
        bearing_std_rad=float(np.deg2rad(float(noise["bearing_std_deg"]))),
    )
    config = DatasetBuildConfig(
        master_seed=int(raw["master_seed"]),
        mobility=train_mobility,
        ood_mobility=ood_mobility,
        observations=observations,
    )
    return LoadedSyntheticProtocol(
        build_config=config,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )
