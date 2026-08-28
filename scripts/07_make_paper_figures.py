from __future__ import annotations

import argparse

from predictive_pc_fmcw.paper_artifacts import (
    make_example_motion_figure,
    make_paper_figures,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/paper_run")
    parser.add_argument(
        "--womd-export", default="data/example/womd_trajectories.json"
    )
    args = parser.parse_args()
    root = args.run_dir
    print(
        make_paper_figures(
            f"{root}/motion_baselines/forecast_summary.json",
            f"{root}/paper_ablations/paper_ablation_summary.json",
            f"{root}/figures",
        )
    )
    print(
        make_example_motion_figure(
            args.womd_export, f"{root}/figures/example_womd_motion.png"
        )
    )


if __name__ == "__main__":
    main()
