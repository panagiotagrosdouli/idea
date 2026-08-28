from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike


def jains_fairness(values: ArrayLike) -> float:
    data = np.asarray(values, dtype=np.float64)
    denominator = data.size * float(np.square(data).sum())
    if denominator <= 0:
        return 1.0
    return float(data.sum() ** 2 / denominator)


@dataclass(frozen=True)
class SimulationMetrics:
    scheduler: str
    scenario_id: str
    source: str
    seed: int
    vehicles: int
    duration_s: float
    generated_packets: int
    delivered_packets: int
    failed_attempts: int
    deadline_dropped_packets: int
    overflow_dropped_packets: int
    remaining_packets: int
    goodput_mbps: float
    packet_delivery_ratio: float
    scheduled_outage_fraction: float
    availability_outage_fraction: float
    mean_latency_ms: float
    p95_latency_ms: float
    deadline_miss_ratio: float
    delivered_before_expiry_ratio: float
    undelivered_packets_at_disconnect: int
    jain_fairness: float
    switch_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    low: float
    high: float
    samples: int


def bootstrap_mean_ci(
    values: Iterable[float],
    samples: int = 2_000,
    seed: int = 20260827,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return ConfidenceInterval(float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, data.size, size=(samples, data.size))
    bootstrap = data[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return ConfidenceInterval(
        mean=float(data.mean()),
        low=float(np.quantile(bootstrap, alpha)),
        high=float(np.quantile(bootstrap, 1 - alpha)),
        samples=int(data.size),
    )


def paired_bootstrap_difference(
    proposed: ArrayLike,
    baseline: ArrayLike,
    samples: int = 2_000,
    seed: int = 20260827,
) -> ConfidenceInterval:
    first = np.asarray(proposed, dtype=np.float64)
    second = np.asarray(baseline, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("Paired arrays must have identical shapes.")
    return bootstrap_mean_ci(first - second, samples=samples, seed=seed)
