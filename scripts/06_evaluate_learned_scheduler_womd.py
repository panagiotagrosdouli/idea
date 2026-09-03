from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from predictive_pc_fmcw.benchmark import (
    run_scenario_benchmark,
    write_benchmark_artifacts,
)
from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.data.womd_official import load_official_womd_tfrecords
from predictive_pc_fmcw.learning.inference import TorchCheckpointPredictor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run packet-level learned scheduling on untouched official WOMD "
            "validation scenarios with paired traffic and channel randomness."
        )
    )
    parser.add_argument("tfrecords", nargs="+")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output", default="artifacts/learned_scheduler_womd")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--max-vehicles", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=[
            "reactive_greedy",
            "proportional_fair",
            "cv_predictive",
            "kalman_predictive",
            "imm_predictive",
            "link_lifetime",
            "learned_predictive",
            "oracle",
        ],
    )
    parser.add_argument(
        "--traffic-seeds",
        nargs="+",
        type=int,
        default=[20260827, 20260828, 20260829, 20260830, 20260831],
    )
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    scenarios = load_official_womd_tfrecords(
        args.tfrecords,
        max_scenarios=args.max_scenarios,
        max_vehicles=args.max_vehicles,
    )
    if not scenarios:
        raise ValueError("No valid official WOMD validation scenarios were loaded.")
    config = load_config(args.config)
    manifest = {
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "scenario_count": len(scenarios),
        "tfrecords": [str(path) for path in args.tfrecords],
        "checkpoints": args.checkpoints,
        "paired_schedulers": args.schedulers,
        "traffic_seeds": args.traffic_seeds,
        "independent_statistical_unit": "scenario_id",
    }
    (destination / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    completed = []
    for checkpoint in args.checkpoints:
        predictor = TorchCheckpointPredictor(checkpoint, device=args.device)
        payload = predictor._torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        training = payload.get("training", {})
        objective = str(training.get("objective", "unknown"))
        seed = int(training.get("seed", -1))
        run_dir = destination / objective / f"seed_{seed}"
        summary_path = run_dir / "summary.json"
        if summary_path.is_file():
            completed.append(str(summary_path))
            continue
        outputs = []
        for traffic_seed in args.traffic_seeds:
            outputs.extend(
                run_scenario_benchmark(
                    scenarios,
                    replace(config, seed=traffic_seed),
                    scheduler_names=args.schedulers,
                    learned_predictor=predictor,
                )
            )
        artifacts = write_benchmark_artifacts(outputs, config, run_dir)
        completed.append(str(artifacts["summary"]))
        (destination / "completed_runs.json").write_text(
            json.dumps(completed, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"completed": completed, "count": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
