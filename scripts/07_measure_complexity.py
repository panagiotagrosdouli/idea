from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.complexity import (
    measure_complexity,
    write_complexity_artifacts,
)
from predictive_pc_fmcw.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="artifacts/corrected_v2/complexity")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    artifacts = write_complexity_artifacts(
        measure_complexity(load_config(args.config), repeats=args.repeats),
        args.output,
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
