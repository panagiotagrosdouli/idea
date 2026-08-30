from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
)
from predictive_pc_fmcw.probabilistic import (
    evaluate_probabilistic_baselines,
    write_probabilistic_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scenario-safe Gaussian calibration for classical baselines."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--output", default="artifacts/corrected_v2/probabilistic"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    slots = (
        max(2, int(round(config.benchmark.duration_s / config.slot_duration_s)))
        if config.benchmark.duration_s is not None
        else config.benchmark.slots
    )
    scenarios = [
        generate_synthetic_scenario(
            config.seed + episode,
            slots=slots,
            vehicles=config.benchmark.vehicles,
            dt_s=config.slot_duration_s,
        )
        for episode in range(config.benchmark.episodes)
    ]
    split = max(1, len(scenarios) // 2)
    calibration, rows = evaluate_probabilistic_baselines(
        scenarios[:split],
        scenarios[split:],
        {
            "gaussian_cv": ConstantVelocityPredictor(),
            "gaussian_ca": ConstantAccelerationPredictor(),
        },
        config.prediction_horizon_steps,
    )
    artifacts = write_probabilistic_artifacts(
        calibration, rows, args.output
    )
    print(json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2))


if __name__ == "__main__":
    main()
