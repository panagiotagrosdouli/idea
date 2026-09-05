#!/usr/bin/env python3
"""Build episode-level operating-region gain and heatmap inputs."""

from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.synthetic.operating_region import analyze_operating_region


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="artifacts/synthetic_dataset_v1")
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze_operating_region(
        args.dataset,
        args.sweep_dir,
        args.output,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "heatmap_rows": len(report["heatmap_rows"]),
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
