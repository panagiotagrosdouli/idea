from __future__ import annotations

import argparse

import numpy as np

from predictive_pc_fmcw.ber import simulate_dbpsk_ber_adaptive, write_ber_lut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/ber/dbpsk_ber_lut.csv")
    parser.add_argument("--bits", type=int, default=250_000)
    parser.add_argument("--max-bits", type=int, default=1_000_000)
    parser.add_argument("--target-errors", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    points = simulate_dbpsk_ber_adaptive(
        np.arange(-5.0, 26.0),
        min_bits=args.bits,
        max_bits=args.max_bits,
        target_errors=args.target_errors,
        seed=args.seed,
    )
    print(write_ber_lut(points, args.output))


if __name__ == "__main__":
    main()
