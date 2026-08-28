from __future__ import annotations

import argparse

from predictive_pc_fmcw.cli import main as cli_main


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="artifacts/paper_ablations")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    command = [
        "paper-ablation",
        "--config",
        args.config,
        "--output",
        args.output,
    ]
    if args.quick:
        command.append("--quick")
    raise SystemExit(cli_main(command))


if __name__ == "__main__":
    main()
