from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .ber import BERPoint, simulate_part_a_notebook_receiver_ber

Estimator = Callable[..., list[BERPoint]]


def paired_chirp_reversal_diagnostic(
    lower_snr_db: float,
    higher_snr_db: float,
    *,
    trials: int = 100,
    bootstrap_repetitions: int = 10_000,
    seed: int = 20260827,
    decisions_per_trial: int = 9_999,
    estimator: Estimator = simulate_part_a_notebook_receiver_ber,
) -> dict[str, object]:
    """Test an adjacent BER reversal using paired independent chirps.

    Within a trial, both SNR calls use the same payload and standardized noise.
    Independent chirps, rather than individual bits, are the sampling units.
    """
    if higher_snr_db <= lower_snr_db:
        raise ValueError("higher_snr_db must exceed lower_snr_db")
    if trials < 2 or bootstrap_repetitions < 100:
        raise ValueError("need at least two trials and 100 bootstrap repetitions")
    if decisions_per_trial < 1_000:
        raise ValueError("decisions_per_trial must be at least 1000")

    trial_seeds = np.random.SeedSequence(seed).generate_state(trials)
    lower, higher = [], []
    for trial_seed in trial_seeds:
        kwargs = {"bits": decisions_per_trial, "seed": int(trial_seed)}
        lower.append(float(estimator([lower_snr_db], **kwargs)[0].simulated_ber))
        higher.append(float(estimator([higher_snr_db], **kwargs)[0].simulated_ber))
    lower_values = np.asarray(lower)
    higher_values = np.asarray(higher)
    delta = higher_values - lower_values

    rng = np.random.default_rng(seed + 1)
    indices = rng.integers(0, trials, size=(bootstrap_repetitions, trials))
    boot = delta[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return {
        "schema": "paired_chirp_ber_reversal_v1",
        "lower_snr_db": float(lower_snr_db),
        "higher_snr_db": float(higher_snr_db),
        "trials": trials,
        "decisions_per_trial": decisions_per_trial,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "sampling_unit": "independent_chirp",
        "common_random_numbers_within_pair": True,
        "mean_lower_ber": float(lower_values.mean()),
        "mean_higher_ber": float(higher_values.mean()),
        "mean_paired_delta_higher_minus_lower": float(delta.mean()),
        "paired_delta_ci95": [float(ci_low), float(ci_high)],
        "higher_snr_worse_supported": bool(ci_low > 0.0),
        "material_reversal_supported": bool(ci_low >= 0.01),
        "catastrophic_trial_threshold_ber": 0.05,
        "lower_catastrophic_trials": int(np.sum(lower_values >= 0.05)),
        "higher_catastrophic_trials": int(np.sum(higher_values >= 0.05)),
        "trial_lower_ber": lower_values.tolist(),
        "trial_higher_ber": higher_values.tolist(),
    }


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
    """Measure receiver stability across independently seeded one-chirp trials.

    Each receiver call consumes at most one partial Part-A chirp. Inference is
    performed over independent chirp realizations instead of correlated bits.
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

    children = np.random.SeedSequence(seed).spawn(
        grid.size * trials_per_snr + 1
    )
    bootstrap_rng = np.random.default_rng(children[-1])
    rows: list[dict[str, Any]] = []
    cursor = 0
    for snr in grid:
        trial_rows: list[dict[str, Any]] = []
        rates: list[float] = []
        for trial_index in range(trials_per_snr):
            trial_seed = int(children[cursor].generate_state(1)[0])
            cursor += 1
            point = receiver(
                [float(snr)], bits=decisions_per_trial, seed=trial_seed
            )[0]
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
            values,
            rng=bootstrap_rng,
            resamples=bootstrap_resamples,
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


def write_diagnostic(report: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination
