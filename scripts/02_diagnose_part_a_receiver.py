from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.ber_diagnostics import run_chirp_cluster_diagnostic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Part-A BER variability across independent chirps."
    )
    parser.add_argument(
        "--snr-db", nargs="+", type=float, default=[5.0, 7.0, 8.0, 10.0]
    )
    parser.add_argument("--trials-per-snr", type=int, default=50)
    parser.add_argument("--decisions-per-trial", type=int, default=1_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--catastrophic-ber-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/02_link/chirp_cluster_diagnostic.json",
    )
    args = parser.parse_args()
    report = run_chirp_cluster_diagnostic(
        np.asarray(args.snr_db, dtype=np.float64),
        trials_per_snr=args.trials_per_snr,
        decisions_per_trial=args.decisions_per_trial,
        bootstrap_resamples=args.bootstrap_resamples,
        catastrophic_ber_threshold=args.catastrophic_ber_threshold,
        seed=args.seed,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(destination)


if __name__ == "__main__":
    main()
