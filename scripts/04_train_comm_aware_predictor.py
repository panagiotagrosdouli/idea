from __future__ import annotations

import argparse

from predictive_pc_fmcw.learning.train import train_from_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--output", default="checkpoints/communication_aware")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lambda-link", type=float, default=0.2)
    parser.add_argument("--lambda-outage", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()
    result = train_from_npz(
        args.dataset,
        args.output,
        epochs=args.epochs,
        lambda_link=args.lambda_link,
        lambda_outage=args.lambda_outage,
        seed=args.seed,
    )
    print(result)


if __name__ == "__main__":
    main()
