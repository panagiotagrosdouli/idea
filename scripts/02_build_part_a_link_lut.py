from __future__ import annotations

import argparse

import numpy as np

from predictive_pc_fmcw.ber import simulate_part_a_notebook_receiver_ber, write_ber_lut
from predictive_pc_fmcw.link_verification import CANONICAL_MIN_BITS_PER_POINT

CANONICAL_SEED = 20260827
CANONICAL_SNR_GRID_DB = np.arange(-5.0, 26.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the canonical Part-A FFT/DPSK BER lookup table."
    )
    parser.add_argument(
        "--output", default="artifacts/paper_final/02_link/dbpsk_ber_lut.csv"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=CANONICAL_MIN_BITS_PER_POINT,
        help="Bits evaluated at every SNR point; canonical minimum is 250000.",
    )
    parser.add_argument("--seed", type=int, default=CANONICAL_SEED)
    args = parser.parse_args()

    if args.bits < CANONICAL_MIN_BITS_PER_POINT:
        parser.error(
            "canonical Part-A LUT requires at least "
            f"{CANONICAL_MIN_BITS_PER_POINT} bits per SNR point"
        )

    points = simulate_part_a_notebook_receiver_ber(
        CANONICAL_SNR_GRID_DB,
        bits=args.bits,
        seed=args.seed,
    )
    print(write_ber_lut(points, args.output))


if __name__ == "__main__":
    main()
