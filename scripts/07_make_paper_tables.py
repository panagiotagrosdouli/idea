from __future__ import annotations

import argparse

from predictive_pc_fmcw.paper_artifacts import make_paper_tables


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/paper_run")
    args = parser.parse_args()
    root = args.run_dir
    print(
        make_paper_tables(
            f"{root}/synthetic_benchmark/summary.json",
            f"{root}/motion_baselines/forecast_summary.json",
            f"{root}/paper_ablations/paper_ablation_summary.json",
            f"{root}/paper_tables.tex",
        )
    )


if __name__ == "__main__":
    main()
