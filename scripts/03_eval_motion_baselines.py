from __future__ import annotations

import argparse

from predictive_pc_fmcw.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="artifacts/motion_baselines")
    parser.add_argument("--womd-export")
    parser.add_argument("--anchor-stride", type=int, default=5)
    args = parser.parse_args()
    command = [
        "motion-eval",
        "--config",
        args.config,
        "--output",
        args.output,
        "--anchor-stride",
        str(args.anchor_stride),
    ]
    if args.womd_export:
        command.extend(["--womd-export", args.womd_export])
    raise SystemExit(cli_main(command))


if __name__ == "__main__":
    main()
