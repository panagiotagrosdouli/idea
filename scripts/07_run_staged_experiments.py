from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.staged_experiments import (
    run_staged_experiments,
    write_staged_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deconfounded operating-region and robustness studies."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="artifacts/staged_experiments")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[20260827, 20260828, 20260829, 20260830, 20260831],
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rows = run_staged_experiments(
        load_config(args.config), tuple(args.seeds), quick=args.quick
    )
    artifacts = write_staged_artifacts(rows, args.output)
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
