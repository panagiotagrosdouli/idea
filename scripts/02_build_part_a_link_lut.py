from __future__ import annotations

import argparse

import numpy as np

from predictive_pc_fmcw.ber import simulate_dbpsk_ber, write_ber_lut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/ber/dbpsk_ber_lut.csv")
    parser.add_argument("--bits", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    points = simulate_dbpsk_ber(
        np.arange(-4.0, 17.0), bits=args.bits, seed=args.seed
    )
    print(write_ber_lut(points, args.output))


if __name__ == "__main__":
    main()
