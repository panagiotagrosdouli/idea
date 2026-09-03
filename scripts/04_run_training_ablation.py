from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.learning.ablation import (
    build_training_ablation_plan,
    run_training_ablation,
)

CANONICAL_SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered four-objective, five-seed GRU ablation."
    )
    parser.add_argument("dataset")
    parser.add_argument("--output", default="artifacts/learned_ablation")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS)
    parser.add_argument("--lambda-link", type=float, default=0.2)
    parser.add_argument("--lambda-outage", type=float, default=0.1)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate and print the immutable run plan without importing PyTorch.",
    )
    args = parser.parse_args()
    seeds = tuple(args.seeds)
    if args.plan_only:
        plan = build_training_ablation_plan(args.dataset, seeds, args.epochs)
        print(json.dumps(asdict(plan), indent=2))
        return
    results = run_training_ablation(
        args.dataset,
        args.output,
        link_config=load_config(args.config).link,
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lambda_link=args.lambda_link,
        lambda_outage=args.lambda_outage,
    )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
