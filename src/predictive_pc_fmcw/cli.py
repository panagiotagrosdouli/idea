from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .benchmark import (
    run_horizon_ablation,
    run_scenario_benchmark,
    run_synthetic_benchmark,
    write_benchmark_artifacts,
)
from .ber import simulate_dbpsk_ber, write_ber_lut
from .config import load_config
from .data.training_export import build_relative_motion_training_npz
from .data.womd_export import load_womd_motion_scenarios
from .experiment_matrix import load_matrix, run_experiment_matrix, write_matrix
from .validation import run_validation


def _write_ablation(rows: list[dict[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "horizon_ablation.json").write_text(
        json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
    )
    with (output / "horizon_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted({int(row["prediction_horizon_steps"]) for row in rows})
    schedulers = sorted({str(row["scheduler"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for scheduler in schedulers:
        goodput = []
        outage = []
        for horizon in horizons:
            subset = [
                row
                for row in rows
                if row["scheduler"] == scheduler
                and row["prediction_horizon_steps"] == horizon
            ]
            goodput.append(np.mean([row["goodput_mbps"] for row in subset]))
            outage.append(
                np.mean([row["scheduled_outage_fraction"] for row in subset])
            )
        axes[0].plot(horizons, goodput, marker="o", label=scheduler)
        axes[1].plot(horizons, outage, marker="o", label=scheduler)
    axes[0].set_ylabel("Goodput (Mbps)")
    axes[1].set_ylabel("Scheduled outage")
    for axis in axes:
        axis.set_xlabel("Prediction horizon (steps)")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "horizon_ablation.png", dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcfmcw", description="Predictive PC-FMCW/DPSK research pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ber = subparsers.add_parser("ber-lut", help="Generate a DBPSK BER-vs-SNR LUT")
    ber.add_argument("--output", default="results/ber/dbpsk_ber_lut.csv")
    ber.add_argument("--bits", type=int, default=250_000)
    ber.add_argument("--seed", type=int, default=20260827)
    ber.add_argument("--snr-min", type=float, default=-4.0)
    ber.add_argument("--snr-max", type=float, default=16.0)
    ber.add_argument("--snr-step", type=float, default=1.0)

    benchmark = subparsers.add_parser("benchmark", help="Run scheduler benchmark")
    benchmark.add_argument("--config", default="configs/default.json")
    benchmark.add_argument("--output", default="results/benchmark")
    benchmark.add_argument("--womd-export")
    benchmark.add_argument("--max-vehicles", type=int)
    benchmark.add_argument("--checkpoint")

    ablation = subparsers.add_parser("ablation", help="Run horizon ablation")
    ablation.add_argument("--config", default="configs/default.json")
    ablation.add_argument("--output", default="results/ablation")
    ablation.add_argument("--horizons", nargs="+", type=int, default=[3, 5, 10, 20])

    prepare = subparsers.add_parser(
        "prepare-training", help="Build a scenario-safe relative-motion NPZ"
    )
    prepare.add_argument("womd_export")
    prepare.add_argument("output")
    prepare.add_argument("--max-vehicles", type=int)

    validate = subparsers.add_parser("validate", help="Run scientific sanity gates")
    validate.add_argument("--config", default="configs/default.json")
    validate.add_argument("--output", default="results/validation.json")

    train = subparsers.add_parser("train", help="Train communication-aware GRU")
    train.add_argument("dataset")
    train.add_argument("output")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--lambda-link", type=float, default=0.2)
    train.add_argument("--lambda-outage", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=20260827)

    matrix = subparsers.add_parser("matrix", help="Run the scripted experiment matrix")
    matrix.add_argument("--config", default="configs/default.json")
    matrix.add_argument("--matrix", default="configs/experiment_matrix.json")
    matrix.add_argument("--output", default="results/matrix")
    matrix.add_argument(
        "--quick",
        action="store_true",
        help="Use the first two values of each axis for a fast integration run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ber-lut":
        grid = np.arange(
            args.snr_min,
            args.snr_max + 0.5 * args.snr_step,
            args.snr_step,
        )
        path = write_ber_lut(
            simulate_dbpsk_ber(grid, bits=args.bits, seed=args.seed), args.output
        )
        print(path)
        return 0
    if args.command == "benchmark":
        config = load_config(args.config)
        learned_predictor = None
        scheduler_names = config.benchmark.schedulers
        if args.checkpoint:
            from .learning.inference import TorchCheckpointPredictor

            learned_predictor = TorchCheckpointPredictor(args.checkpoint)
            scheduler_names = (*scheduler_names, "learned_predictive")
        if args.womd_export:
            scenarios = load_womd_motion_scenarios(
                args.womd_export, max_vehicles=args.max_vehicles
            )
            outputs = run_scenario_benchmark(
                scenarios,
                config,
                scheduler_names=scheduler_names,
                learned_predictor=learned_predictor,
            )
        else:
            outputs = run_synthetic_benchmark(
                config,
                scheduler_names=scheduler_names,
                learned_predictor=learned_predictor,
            )
        artifacts = write_benchmark_artifacts(outputs, config, args.output)
        printable = {key: str(value) for key, value in artifacts.items()}
        print(json.dumps(printable, indent=2))
        return 0
    if args.command == "ablation":
        config = load_config(args.config)
        rows = run_horizon_ablation(config, args.horizons)
        _write_ablation(rows, Path(args.output))
        print(args.output)
        return 0
    if args.command == "prepare-training":
        print(
            build_relative_motion_training_npz(
                args.womd_export, args.output, max_vehicles=args.max_vehicles
            )
        )
        return 0
    if args.command == "validate":
        report = run_validation(load_config(args.config))
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(json.dumps(report.to_dict(), indent=2))
        return 0 if report.status == "PASS" else 1
    if args.command == "train":
        from .learning.train import train_from_npz

        result = train_from_npz(
            args.dataset,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lambda_link=args.lambda_link,
            lambda_outage=args.lambda_outage,
            seed=args.seed,
        )
        print(json.dumps(result.__dict__, indent=2))
        return 0
    if args.command == "matrix":
        config = load_config(args.config)
        matrix_config = load_matrix(args.matrix)
        if args.quick:
            matrix_config = {
                key: values[:2] if len(values) > 1 else values
                for key, values in matrix_config.items()
            }
        artifacts = write_matrix(
            run_experiment_matrix(config, matrix_config), args.output
        )
        printable = {key: str(value) for key, value in artifacts.items()}
        print(json.dumps(printable, indent=2))
        return 0
    raise AssertionError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
