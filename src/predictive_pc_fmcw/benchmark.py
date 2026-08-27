from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data.scenario import MotionScenario
from .data.synthetic import generate_synthetic_scenario
from .link import LinkModel
from .metrics import bootstrap_mean_ci, paired_bootstrap_difference
from .predictors import TrajectoryPredictor
from .simulation.engine import SimulationOutput, run_simulation
from .traffic import generate_traffic_trace

PRIMARY_METRICS = (
    "goodput_mbps",
    "packet_delivery_ratio",
    "scheduled_outage_fraction",
    "mean_latency_ms",
    "p95_latency_ms",
    "deadline_miss_ratio",
    "jain_fairness",
)


def run_scenario_benchmark(
    scenarios: Iterable[MotionScenario],
    config: ExperimentConfig,
    scheduler_names: Iterable[str] | None = None,
    learned_predictor: TrajectoryPredictor | None = None,
) -> list[SimulationOutput]:
    schedulers = tuple(scheduler_names or config.benchmark.schedulers)
    capacity = LinkModel(config.link).capacity_packets(config.slot_duration_s)
    outputs: list[SimulationOutput] = []
    for episode, scenario in enumerate(scenarios):
        seed = config.seed + episode
        slots = min(config.benchmark.slots, scenario.evaluation_slots)
        traffic = generate_traffic_trace(
            seed=seed + 100_000,
            slots=slots,
            vehicles=scenario.vehicle_count,
            nominal_capacity_packets=capacity,
            config=config.traffic,
        )
        for scheduler_name in schedulers:
            outputs.append(
                run_simulation(
                    scenario=scenario,
                    scheduler_name=scheduler_name,
                    traffic=traffic,
                    config=config,
                    seed=seed,
                    learned_predictor=learned_predictor,
                )
            )
    return outputs


def run_synthetic_benchmark(
    config: ExperimentConfig,
    scheduler_names: Iterable[str] | None = None,
    learned_predictor: TrajectoryPredictor | None = None,
) -> list[SimulationOutput]:
    scenarios = [
        generate_synthetic_scenario(
            seed=config.seed + episode,
            slots=config.benchmark.slots,
            vehicles=config.benchmark.vehicles,
            dt_s=config.slot_duration_s,
        )
        for episode in range(config.benchmark.episodes)
    ]
    return run_scenario_benchmark(
        scenarios,
        config,
        scheduler_names=scheduler_names,
        learned_predictor=learned_predictor,
    )


def summarize_outputs(
    outputs: list[SimulationOutput], config: ExperimentConfig
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for output in outputs:
        grouped.setdefault(output.metrics.scheduler, []).append(
            output.metrics.to_dict()
        )
    summary: dict[str, object] = {
        "config": config.to_dict(),
        "episodes": len({output.metrics.scenario_id for output in outputs}),
        "schedulers": {},
    }
    scheduler_summary: dict[str, object] = {}
    for scheduler, rows in sorted(grouped.items()):
        metric_summary: dict[str, object] = {}
        for metric in PRIMARY_METRICS:
            ci = bootstrap_mean_ci(
                [float(row[metric]) for row in rows],
                samples=config.benchmark.bootstrap_samples,
                seed=config.seed,
            )
            metric_summary[metric] = asdict(ci)
        scheduler_summary[scheduler] = metric_summary
    summary["schedulers"] = scheduler_summary

    baseline = grouped.get("reactive_greedy")
    comparisons: dict[str, object] = {}
    if baseline is not None:
        baseline_by_scenario = {
            str(row["scenario_id"]): row for row in baseline
        }
        for scheduler, rows in grouped.items():
            if scheduler == "reactive_greedy":
                continue
            current_by_scenario = {str(row["scenario_id"]): row for row in rows}
            shared = sorted(set(baseline_by_scenario) & set(current_by_scenario))
            comparisons[scheduler] = {}
            for metric in PRIMARY_METRICS:
                proposed = [float(current_by_scenario[key][metric]) for key in shared]
                reference = [float(baseline_by_scenario[key][metric]) for key in shared]
                comparisons[scheduler][metric] = asdict(
                    paired_bootstrap_difference(
                        proposed,
                        reference,
                        samples=config.benchmark.bootstrap_samples,
                        seed=config.seed,
                    )
                )
    summary["paired_difference_vs_reactive"] = comparisons
    return summary


def write_benchmark_artifacts(
    outputs: list[SimulationOutput],
    config: ExperimentConfig,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [output.metrics.to_dict() for output in outputs]
    raw_json = destination / "episode_metrics.json"
    raw_json.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    summary = summarize_outputs(outputs, config)
    summary_json = destination / "summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    csv_path = destination / "episode_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    latex_path = destination / "main_results.tex"
    latex_path.write_text(_latex_table(summary), encoding="utf-8")
    figure_path = destination / "main_metrics.png"
    _plot_summary(summary, figure_path)
    return {
        "episode_metrics": raw_json,
        "summary": summary_json,
        "csv": csv_path,
        "latex": latex_path,
        "figure": figure_path,
    }


def _latex_table(summary: dict[str, object]) -> str:
    schedulers = summary["schedulers"]
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Policy & Goodput (Mbps) & PDR & Outage & P95 (ms) & Jain \\",
        r"\midrule",
    ]
    for name, metrics in schedulers.items():
        label = name.replace("_", r"\_")
        lines.append(
            f"{label} & "
            f"{metrics['goodput_mbps']['mean']:.3f} & "
            f"{metrics['packet_delivery_ratio']['mean']:.3f} & "
            f"{metrics['scheduled_outage_fraction']['mean']:.3f} & "
            f"{metrics['p95_latency_ms']['mean']:.1f} & "
            f"{metrics['jain_fairness']['mean']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _plot_summary(summary: dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    schedulers = summary["schedulers"]
    names = list(schedulers)
    short = [
        name.replace("predictive", "pred.").replace("proportional", "prop.")
        for name in names
    ]
    panels = [
        ("goodput_mbps", "Goodput (Mbps)", False),
        ("packet_delivery_ratio", "Packet delivery ratio", False),
        ("scheduled_outage_fraction", "Scheduled outage", True),
        ("p95_latency_ms", "P95 latency (ms)", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (metric, label, lower_is_better) in zip(axes.flat, panels, strict=True):
        means = np.asarray([schedulers[name][metric]["mean"] for name in names])
        lows = np.asarray([schedulers[name][metric]["low"] for name in names])
        highs = np.asarray([schedulers[name][metric]["high"] for name in names])
        errors = np.vstack([means - lows, highs - means])
        colors = [
            "#d97706" if name == "reactive_greedy" else "#2563eb"
            for name in names
        ]
        axis.bar(np.arange(len(names)), means, yerr=errors, capsize=3, color=colors)
        axis.set_xticks(np.arange(len(names)), short, rotation=28, ha="right")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
        arrow = "lower is better" if lower_is_better else "higher is better"
        axis.set_title(f"{label} ({arrow})")
    fig.suptitle("Predictive PC-FMCW/DPSK scheduler benchmark", fontsize=15)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run_horizon_ablation(
    config: ExperimentConfig,
    horizons: Iterable[int],
    scheduler_names: tuple[str, ...] = (
        "reactive_greedy",
        "predictive_utility",
        "link_lifetime",
        "oracle",
    ),
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon in horizons:
        benchmark = replace(config.benchmark, schedulers=scheduler_names)
        current = replace(
            config, prediction_horizon_steps=int(horizon), benchmark=benchmark
        )
        outputs = run_synthetic_benchmark(current)
        for output in outputs:
            row = output.metrics.to_dict()
            row["prediction_horizon_steps"] = int(horizon)
            rows.append(row)
    return rows
