from __future__ import annotations

import argparse
from pathlib import Path

from predictive_pc_fmcw.cli import main as cli_main
from predictive_pc_fmcw.paper_artifacts import (
    make_example_motion_figure,
    make_paper_figures,
    make_paper_tables,
)


def _run(arguments: list[str]) -> None:
    status = cli_main(arguments)
    if status != 0:
        raise RuntimeError(f"Pipeline command failed: {arguments}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--matrix", default="configs/paper_experiment_matrix.json"
    )
    parser.add_argument(
        "--womd-export", default="data/example/womd_trajectories.json"
    )
    parser.add_argument("--output", default="artifacts/paper_run")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    ber_lut = root / "ber" / "dbpsk_ber_lut.csv"
    _run(["ber-lut", "--output", str(ber_lut)])
    _run(
        [
            "dataset-manifest",
            args.womd_export,
            str(root / "womd_dataset_manifest.json"),
        ]
    )
    _run(
        [
            "motion-eval",
            "--config",
            args.config,
            "--output",
            str(root / "motion_baselines"),
        ]
    )
    _run(
        [
            "motion-eval",
            "--config",
            args.config,
            "--womd-export",
            args.womd_export,
            "--max-vehicles",
            "5",
            "--anchor-stride",
            "1",
            "--output",
            str(root / "womd_motion_baselines"),
        ]
    )
    _run(
        [
            "benchmark",
            "--config",
            args.config,
            "--output",
            str(root / "synthetic_benchmark"),
        ]
    )
    _run(
        [
            "benchmark",
            "--config",
            args.config,
            "--womd-export",
            args.womd_export,
            "--max-vehicles",
            "5",
            "--output",
            str(root / "womd_proxy_benchmark"),
        ]
    )
    ablation_command = [
        "paper-ablation",
        "--config",
        args.config,
        "--ber-lut",
        str(ber_lut),
        "--output",
        str(root / "paper_ablations"),
    ]
    if args.quick:
        ablation_command.append("--quick")
    _run(ablation_command)
    matrix_command = [
        "matrix",
        "--config",
        args.config,
        "--matrix",
        args.matrix,
        "--output",
        str(root / "experiment_matrix"),
    ]
    if args.quick:
        matrix_command.append("--quick")
    _run(matrix_command)
    tables = make_paper_tables(
        root / "synthetic_benchmark" / "summary.json",
        root / "motion_baselines" / "forecast_summary.json",
        root / "paper_ablations" / "paper_ablation_summary.json",
        root / "paper_tables.tex",
    )
    figures = make_paper_figures(
        root / "motion_baselines" / "forecast_summary.json",
        root / "paper_ablations" / "paper_ablation_summary.json",
        root / "figures",
    )
    example = make_example_motion_figure(
        args.womd_export, root / "figures" / "example_womd_motion.png"
    )
    print({"tables": str(tables), "figures": figures, "example": str(example)})


if __name__ == "__main__":
    main()
