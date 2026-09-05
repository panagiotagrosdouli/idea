from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from predictive_pc_fmcw.benchmark import (
    run_scenario_benchmark,
    write_benchmark_artifacts,
)
from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.data.womd_official import load_official_womd_tfrecords
from predictive_pc_fmcw.learning.completion import verify_completion_manifest
from predictive_pc_fmcw.learning.inference import TorchCheckpointPredictor
from predictive_pc_fmcw.link_verification import verify_lut

CANONICAL_SCHEDULERS = [
    "reactive_greedy",
    "proportional_fair",
    "cv_predictive",
    "kalman_predictive",
    "imm_predictive",
    "link_lifetime",
    "learned_predictive",
    "oracle",
]
CANONICAL_TRAFFIC_SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]


def _heldout_scenario_ids(path: str | Path) -> set[str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "scenario_id" not in reader.fieldnames:
            raise ValueError("Held-out metrics must contain a scenario_id column.")
        return {str(row["scenario_id"]) for row in reader}


def _validate_canonical_args(args: argparse.Namespace) -> None:
    if not args.canonical:
        return
    required = {
        "training_npz": args.training_npz,
        "completion_manifest": args.completion_manifest,
        "ber_lut": args.ber_lut,
        "heldout_metrics": args.heldout_metrics,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Canonical Stage 6 requires: {', '.join(missing)}")
    if len(args.checkpoints) != 20:
        raise ValueError("Canonical Stage 6 requires exactly 20 checkpoints.")
    if args.schedulers != CANONICAL_SCHEDULERS:
        raise ValueError("Canonical Stage 6 scheduler family set was modified.")
    if args.traffic_seeds != CANONICAL_TRAFFIC_SEEDS:
        raise ValueError("Canonical Stage 6 traffic seeds were modified.")
    if args.max_scenarios is not None:
        raise ValueError("Canonical Stage 6 cannot use --max-scenarios.")
    if args.max_vehicles != 16:
        raise ValueError("Canonical Stage 6 requires --max-vehicles 16.")


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
    parser.add_argument("--training-npz")
    parser.add_argument("--completion-manifest")
    parser.add_argument("--ber-lut")
    parser.add_argument("--heldout-metrics")
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Enforce the frozen paper Stage-6 contract.",
    )
    parser.add_argument(
        "--schedulers",
        nargs="+",
        default=CANONICAL_SCHEDULERS,
    )
    parser.add_argument(
        "--traffic-seeds",
        nargs="+",
        type=int,
        default=CANONICAL_TRAFFIC_SEEDS,
    )
    args = parser.parse_args()
    _validate_canonical_args(args)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)

    completion_report = None
    if args.completion_manifest:
        if not args.training_npz:
            raise ValueError(
                "completion-manifest verification requires --training-npz."
            )
        completion_report = verify_completion_manifest(
            args.completion_manifest,
            training_npz=args.training_npz,
            checkpoints=list(args.checkpoints),
        )
        if completion_report["status"] != "PASS":
            raise ValueError("Stage-4 completion manifest verification failed.")

    lut_report = None
    config = load_config(args.config)
    if args.ber_lut:
        lut_report = verify_lut(args.ber_lut)
        if lut_report["status"] != "PASS":
            raise ValueError("The supplied BER LUT does not satisfy the Stage-2 gate.")
        config = replace(
            config,
            link=replace(
                config.link,
                ber_source="lut",
                ber_lut_path=str(Path(args.ber_lut)),
            ),
        )

    scenarios = load_official_womd_tfrecords(
        args.tfrecords,
        max_scenarios=args.max_scenarios,
        max_vehicles=args.max_vehicles,
    )
    if not scenarios:
        raise ValueError("No valid official WOMD validation scenarios were loaded.")
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    heldout_ids = None
    if args.heldout_metrics:
        heldout_ids = _heldout_scenario_ids(args.heldout_metrics)
        if scenario_ids != heldout_ids:
            missing_from_tfrecord = sorted(heldout_ids - scenario_ids)
            extra_in_tfrecord = sorted(scenario_ids - heldout_ids)
            raise ValueError(
                "Raw validation TFRecord scenario set differs from Stage-5 heldout "
                f"evaluation: missing={missing_from_tfrecord[:10]}, "
                f"extra={extra_in_tfrecord[:10]}."
            )

    manifest = {
        "canonical": bool(args.canonical),
        "scenario_ids": sorted(scenario_ids),
        "scenario_count": len(scenarios),
        "tfrecords": [str(path) for path in args.tfrecords],
        "checkpoints": args.checkpoints,
        "completion_verification": completion_report,
        "ber_lut_verification": lut_report,
        "heldout_metrics": args.heldout_metrics,
        "heldout_scenario_count": len(heldout_ids) if heldout_ids is not None else None,
        "paired_schedulers": args.schedulers,
        "traffic_seeds": args.traffic_seeds,
        "independent_statistical_unit": "scenario_id",
        "link_model": {
            "ber_source": config.link.ber_source,
            "ber_lut_path": config.link.ber_lut_path,
            "received_power_calibrated": config.link.received_power_calibrated,
        },
    }
    (destination / "evaluation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
        run_manifest_path = run_dir / "run_manifest.json"
        run_manifest = {
            "checkpoint": str(checkpoint),
            "objective": objective,
            "model_seed": seed,
            "traffic_seeds": args.traffic_seeds,
            "schedulers": args.schedulers,
            "scenario_ids": manifest["scenario_ids"],
            "ber_lut_sha256": (
                lut_report["sha256"] if lut_report is not None else None
            ),
        }
        if summary_path.is_file():
            existing_manifest = (
                json.loads(run_manifest_path.read_text(encoding="utf-8"))
                if run_manifest_path.is_file()
                else None
            )
            if existing_manifest == run_manifest:
                completed.append(str(summary_path))
                continue
            raise RuntimeError(
                f"Existing run {run_dir} was produced by an incompatible plan. "
                "Choose a new output directory or remove that run explicitly."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        run_manifest_path.write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
