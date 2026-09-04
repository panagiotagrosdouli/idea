from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from predictive_pc_fmcw.ber import simulate_part_a_notebook_receiver_ber


def _cluster_bootstrap_interval(
    values: np.ndarray, *, rng: np.random.Generator, resamples: int
) -> tuple[float, float]:
    if values.size < 2:
        raise ValueError("At least two independent chirp trials are required.")
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def run_chirp_cluster_diagnostic(
    snr_db: Sequence[float],
    *,
    trials_per_snr: int = 50,
    decisions_per_trial: int = 1_000,
    bootstrap_resamples: int = 10_000,
    catastrophic_ber_threshold: float = 0.05,
    seed: int = 20260827,
    receiver: Callable[..., list[Any]] = simulate_part_a_notebook_receiver_ber,
) -> dict[str, Any]:
    """Estimate receiver uncertainty across independent one-chirp trials.

    Every receiver call uses a separately spawned seed and consumes only one
    partial chirp. Inference is therefore performed over independent chirp
    realizations instead of treating correlated bit decisions as independent.
    """

    if trials_per_snr < 2:
        raise ValueError("trials_per_snr must be at least 2")
    if decisions_per_trial < 1_000 or decisions_per_trial > 9_999:
        raise ValueError("decisions_per_trial must stay within one Part-A chirp")
    if bootstrap_resamples < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    if not 0.0 < catastrophic_ber_threshold <= 0.5:
        raise ValueError("catastrophic_ber_threshold must be in (0, 0.5]")

    grid = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if grid.size == 0 or not np.isfinite(grid).all():
        raise ValueError("snr_db must contain finite values")

    children = np.random.SeedSequence(seed).spawn(grid.size * trials_per_snr + 1)
    bootstrap_rng = np.random.default_rng(children[-1])
    rows: list[dict[str, Any]] = []
    cursor = 0
    for snr in grid:
        trial_rows: list[dict[str, Any]] = []
        rates: list[float] = []
        for trial_index in range(trials_per_snr):
            trial_seed = int(children[cursor].generate_state(1)[0])
            cursor += 1
            point = receiver([float(snr)], bits=decisions_per_trial, seed=trial_seed)[0]
            rate = float(point.simulated_ber)
            rates.append(rate)
            trial_rows.append(
                {
                    "trial": trial_index,
                    "seed": trial_seed,
                    "decisions": int(point.bits),
                    "errors": int(point.errors),
                    "ber": rate,
                }
            )

        values = np.asarray(rates, dtype=np.float64)
        ci_lower, ci_upper = _cluster_bootstrap_interval(
            values, rng=bootstrap_rng, resamples=bootstrap_resamples
        )
        catastrophic = values >= catastrophic_ber_threshold
        rows.append(
            {
                "snr_db": float(snr),
                "independent_chirp_trials": trials_per_snr,
                "decisions_per_trial": decisions_per_trial,
                "mean_ber": float(values.mean()),
                "median_ber": float(np.median(values)),
                "std_ber_across_chirps": float(values.std(ddof=1)),
                "p95_chirp_ber": float(np.quantile(values, 0.95)),
                "max_chirp_ber": float(values.max()),
                "cluster_bootstrap_ci_95": [ci_lower, ci_upper],
                "catastrophic_chirps": int(catastrophic.sum()),
                "catastrophic_chirp_rate": float(catastrophic.mean()),
                "trials": trial_rows,
            }
        )

    return {
        "method": "independent_one_chirp_trials_with_cluster_bootstrap",
        "snr_semantics": "waveform_sample_snr_db",
        "seed": seed,
        "trials_per_snr": trials_per_snr,
        "decisions_per_trial": decisions_per_trial,
        "bootstrap_resamples": bootstrap_resamples,
        "catastrophic_ber_threshold": catastrophic_ber_threshold,
        "rows": rows,
        "scientific_note": (
            "Bits within a chirp are not treated as independent inference units. "
            "This diagnostic measures receiver stability; it does not validate "
            "the physical-layer model or replace the canonical LUT."
        ),
    }
