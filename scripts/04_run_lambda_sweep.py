from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.learning.lambda_sweep import (
    default_lambda_settings,
    run_lambda_sweep,
)


CANONICAL_SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered communication-loss lambda sweep."
    )
    parser.add_argument("dataset")
    parser.add_argument("--output", default="artifacts/lambda_sweep")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", nargs="+", type=int, default=CANONICAL_SEEDS)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    settings = default_lambda_settings()
    if args.plan_only:
        print(
            json.dumps(
                {
                    "settings": [asdict(setting) for setting in settings],
                    "seeds": args.seeds,
                    "planned_runs": len(settings) * len(args.seeds),
                },
                indent=2,
            )
        )
        return
    results = run_lambda_sweep(
        args.dataset,
        args.output,
        link_config=load_config(args.config).link,
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        settings=settings,
    )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
