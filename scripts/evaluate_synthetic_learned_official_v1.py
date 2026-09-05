#!/usr/bin/env python3
"""Evaluate all 20 frozen learned checkpoints on an official synthetic split."""

from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.synthetic.learned_evaluation import (
    run_official_learned_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/synthetic_dataset_v1")
    parser.add_argument(
        "--training-npz",
        default="artifacts/synthetic_dataset_v1/training_dev.npz",
    )
    parser.add_argument(
        "--ablation",
        default="artifacts/synthetic_dataset_v1/learned_ablation",
    )
    parser.add_argument("--official-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--dt-s", type=float, default=0.1)
    args = parser.parse_args()

    config = load_config(args.config)
    report = run_official_learned_evaluation(
        args.dataset,
        training_npz=args.training_npz,
        ablation_dir=args.ablation,
        official_npz=args.official_npz,
        output_path=args.output,
        link_config=config.link,
        batch_size=args.batch_size,
        dt_s=args.dt_s,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "split": report["split"],
                "objectives": len(report["objectives"]),
                "seeds": len(report["seeds"]),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
