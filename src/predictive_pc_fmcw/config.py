from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LinkConfig:
    data_rate_bps: float = 1e9
    packet_bits: int = 12_000
    resource_fraction: float = 0.005
    reference_distance_m: float = 40.0
    reference_snr_db: float = 18.0
    beam_divergence_half_angle_deg: float = 5.0
    field_of_view_deg: float = 70.0
    pointing_sigma_deg: float = 18.0
    atmospheric_attenuation_per_m: float = 0.004
    outage_ber_threshold: float = 1e-3
    min_received_power_w: float = 1e-15

    def __post_init__(self) -> None:
        if self.data_rate_bps <= 0 or self.packet_bits <= 0:
            raise ValueError("Data rate and packet length must be positive.")
        if not 0 < self.resource_fraction <= 1:
            raise ValueError("resource_fraction must be in (0, 1].")
        if not 0 < self.field_of_view_deg <= 180:
            raise ValueError("field_of_view_deg must be in (0, 180].")
        if not 0 < self.outage_ber_threshold < 0.5:
            raise ValueError("outage_ber_threshold must be in (0, 0.5).")


@dataclass(frozen=True)
class TrafficConfig:
    offered_load: float = 0.72
    deadline_slots: int = 12
    deadline_jitter_slots: int = 4
    max_queue_packets: int = 2_000

    def __post_init__(self) -> None:
        if not 0 <= self.offered_load <= 2:
            raise ValueError("offered_load must be in [0, 2].")
        if self.deadline_slots < 1 or self.max_queue_packets < 1:
            raise ValueError("Deadlines and queue capacity must be positive.")


@dataclass(frozen=True)
class SchedulerConfig:
    goodput_weight: float = 1.0
    outage_weight: float = 1.5
    queue_weight: float = 0.35
    deadline_weight: float = 0.5
    fairness_weight: float = 0.2
    switching_weight: float = 0.02
    opportunity_weight: float = 0.6
    lifetime_weight: float = 2.0


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int = 12
    slots: int = 120
    vehicles: int = 5
    bootstrap_samples: int = 2_000
    schedulers: tuple[str, ...] = (
        "random",
        "round_robin",
        "reactive_greedy",
        "proportional_fair",
        "cv_predictive",
        "predictive_utility",
        "link_lifetime",
        "oracle",
    )


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260827
    slot_duration_s: float = 0.1
    prediction_horizon_steps: int = 10
    discount: float = 0.93
    link: LinkConfig = field(default_factory=LinkConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    def __post_init__(self) -> None:
        if self.slot_duration_s <= 0:
            raise ValueError("slot_duration_s must be positive.")
        if self.prediction_horizon_steps < 1:
            raise ValueError("prediction_horizon_steps must be positive.")
        if not 0 < self.discount <= 1:
            raise ValueError("discount must be in (0, 1].")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple_schedulers(raw: dict[str, Any]) -> dict[str, Any]:
    value = dict(raw)
    if "schedulers" in value:
        value["schedulers"] = tuple(value["schedulers"])
    return value


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        seed=int(raw.get("seed", 20260827)),
        slot_duration_s=float(raw.get("slot_duration_s", 0.1)),
        prediction_horizon_steps=int(raw.get("prediction_horizon_steps", 10)),
        discount=float(raw.get("discount", 0.93)),
        link=LinkConfig(**raw.get("link", {})),
        traffic=TrafficConfig(**raw.get("traffic", {})),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        benchmark=BenchmarkConfig(**_tuple_schedulers(raw.get("benchmark", {}))),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))
