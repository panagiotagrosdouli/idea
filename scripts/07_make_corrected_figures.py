from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.paper_artifacts import make_corrected_result_figures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/corrected_v1")
    args = parser.parse_args()
    root = args.run_dir
    results = make_corrected_result_figures(
        f"{root}/synthetic_benchmark/summary.json",
        f"{root}/synthetic_benchmark/episode_metrics.json",
        f"{root}/staged_experiments/staged_experiment_rows.json",
        f"{root}/scenario_slices/scenario_slices.json",
        f"{root}/figures",
    )
    print(json.dumps({key: str(value) for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
