from __future__ import annotations

import csv
import itertools
import json
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from .benchmark import run_synthetic_benchmark
from .config import ExperimentConfig
from .metrics import bootstrap_mean_ci

MATRIX_SCHEDULERS = (
    "reactive_greedy",
    "cv_predictive",
    "kalman_predictive",
    "imm_predictive",
    "predictive_utility",
    "link_lifetime",
    "oracle",
)


def load_matrix(path: str | Path) -> dict[str, list[Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    required = {"vehicles", "offered_load", "slot_duration_s", "seeds"}
    if not required.issubset(raw):
        raise ValueError(f"Experiment matrix is missing: {required - set(raw)}")
    if not {
        "prediction_horizon_steps",
        "prediction_horizon_seconds",
    }.intersection(raw):
        raise ValueError("Experiment matrix requires a prediction-horizon axis.")
    return raw


def run_experiment_matrix(
    config: ExperimentConfig,
    matrix: dict[str, list[Any]],
    schedulers: tuple[str, ...] = MATRIX_SCHEDULERS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    horizon_key = (
        "prediction_horizon_seconds"
        if "prediction_horizon_seconds" in matrix
        else "prediction_horizon_steps"
    )
    axes = itertools.product(
        matrix[horizon_key],
        matrix["vehicles"],
        matrix["offered_load"],
        matrix["slot_duration_s"],
        matrix["seeds"],
    )
    for horizon_value, vehicles, offered_load, slot_duration_s, seed in axes:
        horizon_steps = (
            max(1, int(round(float(horizon_value) / float(slot_duration_s))))
            if horizon_key == "prediction_horizon_seconds"
            else int(horizon_value)
        )
        horizon_seconds = round(horizon_steps * float(slot_duration_s), 12)
        traffic = replace(config.traffic, offered_load=float(offered_load))
        benchmark = replace(
            config.benchmark,
            episodes=1,
            vehicles=int(vehicles),
            schedulers=schedulers,
        )
        current = replace(
            config,
            seed=int(seed),
            prediction_horizon_steps=horizon_steps,
            slot_duration_s=float(slot_duration_s),
            traffic=traffic,
            benchmark=benchmark,
        )
        for output in run_synthetic_benchmark(current):
            row = output.metrics.to_dict()
            row.update(
                {
                    "prediction_horizon_steps": horizon_steps,
                    "prediction_horizon_s": horizon_seconds,
                    "offered_load": float(offered_load),
                    "slot_duration_s": float(slot_duration_s),
                }
            )
            rows.append(row)
    return rows


def write_matrix(
    rows: list[dict[str, object]], output_dir: str | Path
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No experiment-matrix rows to write.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "experiment_matrix_results.json"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8")
    csv_path = output / "experiment_matrix_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    manifest = output / "matrix_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": len(rows),
                "schedulers": sorted({row["scheduler"] for row in rows}),
                "horizons": sorted(
                    {row["prediction_horizon_s"] for row in rows}
                ),
                "vehicles": sorted({row["vehicles"] for row in rows}),
                "offered_load": sorted({row["offered_load"] for row in rows}),
                "slot_duration_s": sorted({row["slot_duration_s"] for row in rows}),
                "seeds": sorted({row["seed"] for row in rows}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_path = output / "matrix_summary.json"
    summary = _matrix_summary(rows)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figure_path = output / "link_lifetime_gain_vs_load.png"
    _plot_gain_vs_load(summary, figure_path)
    load_figure_path = output / "communication_metrics_vs_load.png"
    _plot_policy_metrics_vs_load(rows, load_figure_path)
    horizon_figure_path = output / "goodput_gain_vs_horizon.png"
    _plot_gain_vs_horizon(rows, horizon_figure_path)
    return {
        "json": json_path,
        "csv": csv_path,
        "manifest": manifest,
        "summary": summary_path,
        "figure": figure_path,
        "load_figure": load_figure_path,
        "horizon_figure": horizon_figure_path,
    }


def _matrix_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    key_fields = (
        "prediction_horizon_s",
        "vehicles",
        "offered_load",
        "slot_duration_s",
        "seed",
    )
    index = {
        tuple(row[field] for field in key_fields) + (row["scheduler"],): row
        for row in rows
    }
    policies = sorted({str(row["scheduler"]) for row in rows})
    comparisons: dict[str, object] = {}
    for policy in policies:
        if policy == "reactive_greedy":
            continue
        differences: list[float] = []
        for key, row in index.items():
            if key[-1] != policy:
                continue
            baseline = index[key[:-1] + ("reactive_greedy",)]
            differences.append(
                float(row["goodput_mbps"]) - float(baseline["goodput_mbps"])
            )
        comparisons[policy] = {
            "paired_points": len(differences),
            "mean_goodput_difference_mbps": statistics.mean(differences),
            "median_goodput_difference_mbps": statistics.median(differences),
            "goodput_win_fraction": sum(value > 0 for value in differences)
            / len(differences),
            **_paired_statistics(differences),
        }
    load_slices: dict[str, object] = {}
    for load in sorted({float(row["offered_load"]) for row in rows}):
        differences = []
        for key, row in index.items():
            if key[-1] == "link_lifetime" and float(row["offered_load"]) == load:
                baseline = index[key[:-1] + ("reactive_greedy",)]
                differences.append(
                    float(row["goodput_mbps"])
                    - float(baseline["goodput_mbps"])
                )
        if not differences:
            continue
        load_slices[str(load)] = {
            "paired_points": len(differences),
            "mean_goodput_difference_mbps": statistics.mean(differences),
            "goodput_win_fraction": sum(value > 0 for value in differences)
            / len(differences),
            "bootstrap_95_ci_mbps": _paired_statistics(differences)[
                "bootstrap_95_ci_mbps"
            ],
        }
    return {
        "paired_comparison_vs_reactive": comparisons,
        "link_lifetime_by_offered_load": load_slices,
    }


def _paired_statistics(differences: list[float]) -> dict[str, object]:
    import numpy as np
    from scipy import stats

    values = np.asarray(differences, dtype=np.float64)
    interval = bootstrap_mean_ci(values, samples=5_000, seed=20260827)
    if values.size > 1:
        t_test_p = float(stats.ttest_1samp(values, 0.0).pvalue)
        standard_deviation = float(values.std(ddof=1))
        cohen_dz = (
            float(values.mean() / standard_deviation)
            if standard_deviation > 0
            else float("nan")
        )
    else:
        t_test_p = float("nan")
        cohen_dz = float("nan")
    nonzero = values[values != 0]
    wilcoxon_p = (
        float(stats.wilcoxon(nonzero).pvalue)
        if nonzero.size > 0
        else float("nan")
    )
    return {
        "bootstrap_95_ci_mbps": {"low": interval.low, "high": interval.high},
        "paired_t_test_p_value": t_test_p,
        "wilcoxon_p_value": wilcoxon_p,
        "cohen_dz": cohen_dz,
    }


def _plot_gain_vs_load(summary: dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slices = summary["link_lifetime_by_offered_load"]
    loads = sorted(float(value) for value in slices)
    gains = [slices[str(load)]["mean_goodput_difference_mbps"] for load in loads]
    wins = [100 * slices[str(load)]["goodput_win_fraction"] for load in loads]
    fig, first = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    second = first.twinx()
    first.plot(loads, gains, color="#2563eb", marker="o", linewidth=2)
    second.plot(loads, wins, color="#d97706", marker="s", linewidth=2)
    first.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    first.set_xlabel("Normalized offered load")
    first.set_ylabel("Mean goodput difference vs reactive (Mbps)", color="#2563eb")
    second.set_ylabel("Goodput win fraction (%)", color="#d97706")
    first.grid(alpha=0.25)
    first.set_title("Link-lifetime scheduler operating region")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_policy_metrics_vs_load(
    rows: list[dict[str, object]], path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = (
        "reactive_greedy",
        "cv_predictive",
        "link_lifetime",
        "oracle",
    )
    loads = sorted({float(row["offered_load"]) for row in rows})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for policy in policies:
        if not any(row["scheduler"] == policy for row in rows):
            continue
        goodput = []
        latency = []
        for load in loads:
            selected = [
                row
                for row in rows
                if row["scheduler"] == policy
                and float(row["offered_load"]) == load
            ]
            goodput.append(
                statistics.mean(float(row["goodput_mbps"]) for row in selected)
            )
            latency.append(
                statistics.mean(float(row["p95_latency_ms"]) for row in selected)
            )
        axes[0].plot(loads, goodput, marker="o", label=policy)
        axes[1].plot(loads, latency, marker="o", label=policy)
    axes[0].set_ylabel("Goodput (Mbps)")
    axes[1].set_ylabel("P95 latency (ms)")
    for axis in axes:
        axis.set_xlabel("Normalized offered load")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.suptitle("Communication performance versus offered load")
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_gain_vs_horizon(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key_fields = (
        "prediction_horizon_s",
        "vehicles",
        "offered_load",
        "slot_duration_s",
        "seed",
    )
    index = {
        tuple(row[field] for field in key_fields) + (row["scheduler"],): row
        for row in rows
    }
    horizons = sorted({float(row["prediction_horizon_s"]) for row in rows})
    gains = []
    for horizon in horizons:
        differences = []
        for key, row in index.items():
            if key[-1] != "link_lifetime" or float(key[0]) != horizon:
                continue
            baseline = index[key[:-1] + ("reactive_greedy",)]
            differences.append(
                float(row["goodput_mbps"])
                - float(baseline["goodput_mbps"])
            )
        gains.append(
            statistics.mean(differences) if differences else float("nan")
        )
    figure, axis = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    axis.plot(horizons, gains, color="#2563eb", marker="o", linewidth=2)
    axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Prediction horizon (s)")
    axis.set_ylabel("Link-lifetime goodput gain vs reactive (Mbps)")
    axis.set_title("Value of prediction horizon")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=220)
    plt.close(figure)
