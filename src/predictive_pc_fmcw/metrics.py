from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

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
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    deadline_miss_ratio: float
    censored_packet_ratio: float
    deadline_or_censored_ratio: float
    delivered_before_expiry_ratio: float
    undelivered_packets_at_disconnect: int
    jain_fairness: float
    demand_normalized_jain_fairness: float
    mean_scheduled_snr_db: float
    mean_scheduled_ber: float
    mean_scheduled_per: float
    mean_scheduled_relative_power: float
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


def holm_adjusted_pvalues(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted.tolist()
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = order.size
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def paired_metric_statistics(
    proposed: ArrayLike,
    baseline: ArrayLike,
    *,
    higher_is_better: bool,
    clusters: Iterable[object] | None = None,
    samples: int = 5_000,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Direction-aware paired inference at the independent-cluster level."""

    from scipy import stats

    first = np.asarray(proposed, dtype=np.float64)
    second = np.asarray(baseline, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Paired inputs must be aligned one-dimensional arrays.")
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if first.size == 0:
        return {
            "pairs": 0,
            "independent_clusters": 0,
            "higher_is_better": higher_is_better,
            "raw_mean_difference": float("nan"),
            "favorable_mean_difference": float("nan"),
            "cluster_mean_favorable_difference": float("nan"),
            "bootstrap_95_ci_favorable": {
                "low": float("nan"),
                "high": float("nan"),
            },
            "favorable_win_fraction": float("nan"),
            "paired_t_test_p_value": float("nan"),
            "wilcoxon_p_value": float("nan"),
            "cohen_dz_favorable": float("nan"),
        }
    raw_difference = first - second
    direction = 1.0 if higher_is_better else -1.0
    favorable = direction * raw_difference
    if clusters is None:
        independent = favorable
        cluster_count = favorable.size
    else:
        labels_all = np.asarray(list(clusters), dtype=object)
        if labels_all.shape != finite.shape:
            raise ValueError("clusters must align with paired inputs.")
        labels = labels_all[finite]
        unique = list(dict.fromkeys(labels.tolist()))
        independent = np.asarray(
            [favorable[labels == label].mean() for label in unique],
            dtype=np.float64,
        )
        cluster_count = len(unique)
    interval = bootstrap_mean_ci(
        independent, samples=samples, seed=seed
    )
    if independent.size > 1:
        standard_deviation = float(independent.std(ddof=1))
        effectively_constant = np.allclose(
            independent,
            independent[0],
            rtol=1e-12,
            atol=1e-12,
        )
        if not effectively_constant:
            t_p = float(stats.ttest_1samp(independent, 0.0).pvalue)
            cohen_dz = float(independent.mean() / standard_deviation)
        else:
            t_p = float("nan")
            cohen_dz = float("nan")
    else:
        t_p = float("nan")
        cohen_dz = float("nan")
    nonzero = independent[independent != 0]
    wilcoxon_p = (
        float(stats.wilcoxon(nonzero).pvalue)
        if nonzero.size > 0
        else float("nan")
    )
    return {
        "pairs": int(first.size),
        "independent_clusters": int(cluster_count),
        "higher_is_better": higher_is_better,
        "raw_mean_difference": float(raw_difference.mean()),
        "favorable_mean_difference": float(favorable.mean()),
        "cluster_mean_favorable_difference": float(independent.mean()),
        "bootstrap_95_ci_favorable": {
            "low": interval.low,
            "high": interval.high,
        },
        "favorable_win_fraction": float(np.mean(favorable > 0)),
        "paired_t_test_p_value": t_p,
        "wilcoxon_p_value": wilcoxon_p,
        "cohen_dz_favorable": cohen_dz,
    }
