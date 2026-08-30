from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.paper_artifacts import (
    make_corrected_result_figures,
    make_probabilistic_calibration_figure,
    make_required_diagnostic_figures,
    make_system_architecture_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="artifacts/corrected_v2")
    args = parser.parse_args()
    root = args.run_dir
    results = make_corrected_result_figures(
        f"{root}/synthetic_benchmark/summary.json",
        f"{root}/synthetic_benchmark/episode_metrics.json",
        f"{root}/staged_experiments/staged_experiment_rows.json",
        f"{root}/scenario_slices/scenario_slices.json",
        f"{root}/figures",
    )
    results.update(
        make_required_diagnostic_figures(
            f"{root}/synthetic_benchmark/summary.json",
            f"{root}/synthetic_benchmark/episode_metrics.json",
            f"{root}/motion_baselines/forecast_metrics.json",
            f"{root}/ber/dbpsk_ber_lut.csv",
            f"{root}/figures",
        )
    )
    results["architecture"] = make_system_architecture_figure(
        f"{root}/figures/system_architecture.png"
    )
    results["probabilistic_calibration"] = (
        make_probabilistic_calibration_figure(
            f"{root}/probabilistic/probabilistic_calibration.json",
            f"{root}/figures/probabilistic_calibration.png",
        )
    )
    print(json.dumps({key: str(value) for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
