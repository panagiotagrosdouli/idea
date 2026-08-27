from __future__ import annotations

import argparse
from pathlib import Path

from predictive_pc_fmcw.cli import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="results/full_pipeline")
    parser.add_argument("--womd-export")
    args = parser.parse_args()
    root = Path(args.output)
    raise SystemExit(
        main(
            [
                "benchmark",
                "--config",
                args.config,
                "--output",
                str(root / ("womd" if args.womd_export else "synthetic")),
                *(["--womd-export", args.womd_export] if args.womd_export else []),
            ]
        )
    )

