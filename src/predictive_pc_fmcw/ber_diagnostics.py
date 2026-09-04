from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

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


def write_diagnostic(report: dict[str, object], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination
