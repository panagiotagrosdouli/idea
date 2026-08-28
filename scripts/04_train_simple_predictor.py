from __future__ import annotations

import argparse

from predictive_pc_fmcw.learning.train import train_from_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="checkpoints/trajectory_only")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    result = train_from_npz(
        args.dataset,
        args.output,
        epochs=args.epochs,
        lambda_link=0.0,
        lambda_outage=0.0,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
