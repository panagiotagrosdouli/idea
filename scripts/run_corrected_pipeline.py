from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.cli import main as cli_main
from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.paper_artifacts import (
    make_corrected_result_figures,
    make_example_motion_figure,
    make_paper_figures,
    make_paper_tables,
)
from predictive_pc_fmcw.staged_experiments import (
    run_staged_experiments,
    write_staged_artifacts,
)


def _run(arguments: list[str]) -> None:
    status = cli_main(arguments)
    if status != 0:
        raise RuntimeError(f"Pipeline command failed: {arguments}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the post-audit corrected research artifacts."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--womd-export", default="data/example/womd_trajectories.json"
    )
    parser.add_argument("--output", default="artifacts/corrected_v1")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    ber_lut = root / "ber" / "dbpsk_ber_lut.csv"
    _run(
        [
            "validate",
            "--config",
            args.config,
            "--output",
            str(root / "validation.json"),
        ]
    )
    _run(
        [
            "ber-lut",
            "--output",
            str(ber_lut),
            "--bits",
            "100000",
            "--max-bits",
            "1000000",
            "--target-errors",
            "200",
        ]
    )
    _run(
        [
            "dataset-manifest",
            args.womd_export,
            str(root / "womd_proxy_dataset_manifest.json"),
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
            "motion-eval",
            "--config",
            args.config,
            "--output",
            str(root / "motion_baselines"),
        ]
    )
    _run(
        [
            "scenario-slices",
            "--config",
            args.config,
            "--output",
            str(root / "scenario_slices"),
        ]
    )
    womd_common = [
        "--config",
        args.config,
        "--womd-export",
        args.womd_export,
        "--max-vehicles",
        "5",
    ]
    _run(
        [
            "benchmark",
            *womd_common,
            "--output",
            str(root / "womd_proxy_benchmark"),
        ]
    )
    _run(
        [
            "motion-eval",
            *womd_common,
            "--anchor-stride",
            "1",
            "--output",
            str(root / "womd_proxy_motion"),
        ]
    )
    _run(
        [
            "scenario-slices",
            *womd_common,
            "--output",
            str(root / "womd_proxy_slices"),
        ]
    )
    ablation = [
        "paper-ablation",
        "--config",
        args.config,
        "--ber-lut",
        str(ber_lut),
        "--output",
        str(root / "paper_ablations"),
    ]
    if args.quick:
        ablation.append("--quick")
    _run(ablation)
    config = load_config(args.config)
    staged_rows = run_staged_experiments(config, quick=args.quick)
    write_staged_artifacts(staged_rows, root / "staged_experiments")
    make_paper_tables(
        root / "synthetic_benchmark" / "summary.json",
        root / "motion_baselines" / "forecast_summary.json",
        root / "paper_ablations" / "paper_ablation_summary.json",
        root / "paper_tables.tex",
    )
    make_paper_figures(
        root / "motion_baselines" / "forecast_summary.json",
        root / "paper_ablations" / "paper_ablation_summary.json",
        root / "figures",
    )
    make_example_motion_figure(
        args.womd_export, root / "figures" / "example_womd_motion.png"
    )
    make_corrected_result_figures(
        root / "synthetic_benchmark" / "summary.json",
        root / "synthetic_benchmark" / "episode_metrics.json",
        root / "staged_experiments" / "staged_experiment_rows.json",
        root / "scenario_slices" / "scenario_slices.json",
        root / "figures",
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "config": args.config,
        "output": str(root),
        "quick_diagnostic": args.quick,
        "official_womd_used": False,
        "learned_checkpoint_used": False,
        "measured_optical_channel_used": False,
        "notes": [
            "Compact WOMD uses proxy ego and model-based communication.",
            "Quick staged inference uses two seeds and is diagnostic only.",
            "Existing artifacts outside this directory are not mixed into this run.",
        ],
    }
    (root / "corrected_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(root)


if __name__ == "__main__":
    main()
