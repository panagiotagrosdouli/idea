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
    channel_mode: str = "full"
    ber_source: str = "analytical"
    ber_lut_path: str | None = None

    def __post_init__(self) -> None:
        if self.data_rate_bps <= 0 or self.packet_bits <= 0:
            raise ValueError("Data rate and packet length must be positive.")
        if not 0 < self.resource_fraction <= 1:
            raise ValueError("resource_fraction must be in (0, 1].")
        if not 0 < self.field_of_view_deg <= 180:
            raise ValueError("field_of_view_deg must be in (0, 180].")
        if not 0 < self.outage_ber_threshold < 0.5:
            raise ValueError("outage_ber_threshold must be in (0, 0.5).")
        if self.channel_mode not in {"range_only", "range_pointing", "full"}:
            raise ValueError(
                "channel_mode must be range_only, range_pointing, or full."
            )
        if self.ber_source not in {"analytical", "lut"}:
            raise ValueError("ber_source must be analytical or lut.")
        if self.ber_source == "lut" and not self.ber_lut_path:
            raise ValueError("ber_lut_path is required when ber_source is lut.")


@dataclass(frozen=True)
class TrafficConfig:
    offered_load: float = 0.72
    deadline_slots: int = 12
    deadline_jitter_slots: int = 4
    max_queue_packets: int = 2_000
    model: str = "poisson"
    periodic_interval_slots: int = 5
    markov_low_rate_scale: float = 0.25
    markov_high_rate_scale: float = 2.5
    markov_low_to_high: float = 0.08
    markov_high_to_low: float = 0.20

    def __post_init__(self) -> None:
        if not 0 <= self.offered_load <= 2:
            raise ValueError("offered_load must be in [0, 2].")
        if self.deadline_slots < 1 or self.max_queue_packets < 1:
            raise ValueError("Deadlines and queue capacity must be positive.")
        if self.model not in {"poisson", "periodic", "markov_modulated"}:
            raise ValueError("Unsupported traffic model.")
        if self.periodic_interval_slots < 1:
            raise ValueError("periodic_interval_slots must be positive.")
        probabilities = (self.markov_low_to_high, self.markov_high_to_low)
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Markov transition probabilities must be in [0, 1].")
        if self.markov_low_rate_scale < 0 or self.markov_high_rate_scale < 0:
            raise ValueError("Markov traffic-rate scales must be non-negative.")


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
        "kalman_predictive",
        "imm_predictive",
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
    history_measurement_noise_std_m: float = 0.0
    forecast_position_noise_std_m: float = 0.0
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
        if self.history_measurement_noise_std_m < 0:
            raise ValueError("history_measurement_noise_std_m must be non-negative.")
        if self.forecast_position_noise_std_m < 0:
            raise ValueError("forecast_position_noise_std_m must be non-negative.")

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
        history_measurement_noise_std_m=float(
            raw.get("history_measurement_noise_std_m", 0.0)
        ),
        forecast_position_noise_std_m=float(
            raw.get("forecast_position_noise_std_m", 0.0)
        ),
        link=LinkConfig(**raw.get("link", {})),
        traffic=TrafficConfig(**raw.get("traffic", {})),
        scheduler=SchedulerConfig(**raw.get("scheduler", {})),
        benchmark=BenchmarkConfig(**_tuple_schedulers(raw.get("benchmark", {}))),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))
