from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.cli import main as cli_main
from predictive_pc_fmcw.complexity import (
    measure_complexity,
    write_complexity_artifacts,
)
from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.data.synthetic import generate_synthetic_scenario
from predictive_pc_fmcw.data.womd_export import load_womd_motion_scenarios
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.paper_artifacts import (
    make_corrected_result_figures,
    make_example_motion_figure,
    make_paper_figures,
    make_paper_tables,
    make_probabilistic_calibration_figure,
    make_required_diagnostic_figures,
    make_scheduler_timeline_figure,
    make_system_architecture_figure,
    make_trajectory_link_trace_figure,
)
from predictive_pc_fmcw.predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
)
from predictive_pc_fmcw.probabilistic import (
    evaluate_probabilistic_baselines,
    write_probabilistic_artifacts,
)
from predictive_pc_fmcw.simulation.engine import run_simulation
from predictive_pc_fmcw.staged_experiments import (
    run_staged_experiments,
    write_staged_artifacts,
)
from predictive_pc_fmcw.traffic import generate_traffic_trace


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
    parser.add_argument("--output", default="artifacts/corrected_v2")
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
            "--receiver",
            "part-a-notebook",
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
    controlled_slots_for_evaluation = (
        max(
            2,
            int(round(config.benchmark.duration_s / config.slot_duration_s)),
        )
        if config.benchmark.duration_s is not None
        else config.benchmark.slots
    )
    calibration_scenarios = [
        generate_synthetic_scenario(
            config.seed + episode,
            slots=controlled_slots_for_evaluation,
            vehicles=config.benchmark.vehicles,
            dt_s=config.slot_duration_s,
        )
        for episode in range(config.benchmark.episodes)
    ]
    calibration_split = max(1, len(calibration_scenarios) // 2)
    gaussian_calibrations, probabilistic_rows = (
        evaluate_probabilistic_baselines(
            calibration_scenarios[:calibration_split],
            calibration_scenarios[calibration_split:],
            {
                "gaussian_cv": ConstantVelocityPredictor(),
                "gaussian_ca": ConstantAccelerationPredictor(),
            },
            config.prediction_horizon_steps,
        )
    )
    write_probabilistic_artifacts(
        gaussian_calibrations,
        probabilistic_rows,
        root / "probabilistic",
    )
    make_probabilistic_calibration_figure(
        root / "probabilistic" / "probabilistic_calibration.json",
        root / "figures" / "probabilistic_calibration.png",
    )
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
    make_required_diagnostic_figures(
        root / "synthetic_benchmark" / "summary.json",
        root / "synthetic_benchmark" / "episode_metrics.json",
        root / "motion_baselines" / "forecast_metrics.json",
        ber_lut,
        root / "figures",
    )
    make_system_architecture_figure(
        root / "figures" / "system_architecture.png"
    )
    controlled_slots = (
        max(
            2,
            int(
                round(
                    config.benchmark.duration_s / config.slot_duration_s
                )
            ),
        )
        if config.benchmark.duration_s is not None
        else config.benchmark.slots
    )
    controlled = generate_synthetic_scenario(
        config.seed,
        slots=controlled_slots,
        vehicles=config.benchmark.vehicles,
        dt_s=config.slot_duration_s,
    )
    link_model = LinkModel(config.link)
    capacity = link_model.capacity_packets(config.slot_duration_s)
    traffic = generate_traffic_trace(
        config.seed + 100_000,
        controlled_slots,
        controlled.vehicle_count,
        capacity,
        config.traffic,
        slot_duration_s=config.slot_duration_s,
    )
    timeline = run_simulation(
        controlled,
        "link_lifetime",
        traffic,
        config,
        seed=config.seed,
    )
    make_scheduler_timeline_figure(
        timeline,
        config.slot_duration_s,
        root / "figures" / "scheduler_timeline.png",
    )
    proxy_scenario = load_womd_motion_scenarios(
        args.womd_export, max_vehicles=5
    )[0]
    make_trajectory_link_trace_figure(
        proxy_scenario,
        link_model,
        root / "figures" / "trajectory_to_link_trace.png",
    )
    write_complexity_artifacts(
        measure_complexity(config), root / "complexity"
    )
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "config": args.config,
        "output": str(root),
        "quick_diagnostic": args.quick,
        "official_womd_used": False,
        "learned_checkpoint_used": False,
        "classical_gaussian_calibration_used": True,
        "measured_optical_channel_used": False,
        "part_a_notebook_receiver_lut_used": True,
        "notes": [
            "Compact WOMD uses proxy ego and model-based communication.",
            (
                "Quick staged inference uses two seeds and is diagnostic only."
                if args.quick
                else "Staged controlled studies use five independent seeds."
            ),
            "Existing artifacts outside this directory are not mixed into this run.",
        ],
    }
    (root / "corrected_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(root)


if __name__ == "__main__":
    main()
