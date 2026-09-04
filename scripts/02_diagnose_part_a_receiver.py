from __future__ import annotations

import argparse

from predictive_pc_fmcw.ber_diagnostics import (
    paired_chirp_reversal_diagnostic,
    write_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired chirp-cluster diagnostic for a raw BER reversal."
    )
    parser.add_argument("--lower-snr-db", type=float, default=7.0)
    parser.add_argument("--higher-snr-db", type=float, default=8.0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/02_link/paired_chirp_7_8_db.json",
    )
    args = parser.parse_args()
    report = paired_chirp_reversal_diagnostic(
        args.lower_snr_db,
        args.higher_snr_db,
        trials=args.trials,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    print(write_diagnostic(report, args.output))


if __name__ == "__main__":
    main()
