from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.learning.lambda_selection import freeze_lambda_selection


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze a development-only lambda choice after a complete sweep."
    )
    parser.add_argument("sweep_dir")
    parser.add_argument("dataset")
    parser.add_argument("--lambda-link", type=float, required=True)
    parser.add_argument("--lambda-outage", type=float, required=True)
    parser.add_argument("--rationale", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/04_learning/lambda_selection.json",
    )
    args = parser.parse_args()
    report = freeze_lambda_selection(
        args.sweep_dir,
        args.dataset,
        lambda_link=args.lambda_link,
        lambda_outage=args.lambda_outage,
        rationale=args.rationale,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
