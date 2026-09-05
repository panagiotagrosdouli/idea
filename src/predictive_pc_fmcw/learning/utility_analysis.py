from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..metrics import holm_adjusted_pvalues, paired_metric_statistics

UTILITY_METRICS = (
    "goodput_mbps",
    "packet_delivery_ratio",
    "scheduled_outage_fraction",
    "p95_latency_ms",
    "deadline_miss_ratio",
    "jain_fairness",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def join_accuracy_and_scheduler_utility(
    heldout_csv: str | Path, scheduler_root: str | Path
) -> list[dict[str, Any]]:
    heldout = _read_csv(Path(heldout_csv))
    accuracy = {
        (row["objective"], int(row["seed"]), row["scenario_id"]): row
        for row in heldout
    }
    joined = []
    pattern = "*/seed_*/episode_metrics.csv"
    for metrics_path in sorted(Path(scheduler_root).glob(pattern)):
        objective = metrics_path.parent.parent.name
        model_seed = int(metrics_path.parent.name.removeprefix("seed_"))
        rows = _read_csv(metrics_path)
        if rows and "seed" not in rows[0]:
            raise ValueError(
                f"{metrics_path} is missing simulation seed required for pairing."
            )
        indexed = {
            (row["scheduler"], row["scenario_id"], int(row["seed"])): row
            for row in rows
        }
        learned_keys = sorted(
            (scenario, simulation_seed)
            for scheduler, scenario, simulation_seed in indexed
            if scheduler == "learned_predictive"
            and ("reactive_greedy", scenario, simulation_seed) in indexed
            and (objective, model_seed, scenario) in accuracy
        )
        for scenario, simulation_seed in learned_keys:
            learned = indexed[("learned_predictive", scenario, simulation_seed)]
            reactive = indexed[("reactive_greedy", scenario, simulation_seed)]
            motion = accuracy[(objective, model_seed, scenario)]
            item: dict[str, Any] = {
                "objective": objective,
                "seed": model_seed,
                "simulation_seed": simulation_seed,
                "scenario_id": scenario,
                "ade_m": float(motion["ade_m"]),
                "fde_m": float(motion["fde_m"]),
            }
            for metric in UTILITY_METRICS:
                proposed = float(learned[metric])
                baseline = float(reactive[metric])
                item[f"learned_{metric}"] = proposed
                item[f"reactive_{metric}"] = baseline
                item[f"delta_{metric}"] = proposed - baseline
            joined.append(item)
    return joined


def _apply_holm_family(metrics: dict[str, dict[str, Any]]) -> None:
    names = list(metrics)
    t_adjusted = holm_adjusted_pvalues(
        metrics[name]["paired_t_test_p_value"] for name in names
    )
    wilcoxon_adjusted = holm_adjusted_pvalues(
        metrics[name]["wilcoxon_p_value"] for name in names
    )
    for name, t_value, wilcoxon_value in zip(
        names, t_adjusted, wilcoxon_adjusted, strict=True
    ):
        metrics[name]["paired_t_test_holm_p_value"] = t_value
        metrics[name]["wilcoxon_holm_p_value"] = wilcoxon_value


def aggregate_scenario_relationship(
    rows: list[dict[str, Any]],
) -> list[dict[str, float | str | int]]:
    """Collapse model/traffic-seed rows to one observation per WOMD scenario."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    aggregated = []
    for scenario_id in sorted(grouped):
        selected = grouped[scenario_id]
        aggregated.append(
            {
                "scenario_id": scenario_id,
                "ade_m": float(np.mean([row["ade_m"] for row in selected])),
                "delta_goodput_mbps": float(
                    np.mean([row["delta_goodput_mbps"] for row in selected])
                ),
                "paired_model_traffic_rows": len(selected),
            }
        )
    return aggregated


def summarize_utility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {}
    for objective in sorted({row["objective"] for row in rows}):
        selected = [row for row in rows if row["objective"] == objective]
        metrics = {}
        for metric in UTILITY_METRICS:
            proposed = np.asarray(
                [row[f"learned_{metric}"] for row in selected], dtype=float
            )
            baseline = np.asarray(
                [row[f"reactive_{metric}"] for row in selected], dtype=float
            )
            metrics[metric] = paired_metric_statistics(
                proposed,
                baseline,
                higher_is_better=metric
                not in {
                    "scheduled_outage_fraction",
                    "p95_latency_ms",
                    "deadline_miss_ratio",
                },
                clusters=[row["scenario_id"] for row in selected],
            )
        _apply_holm_family(metrics)
        summary[objective] = {
            "paired_model_traffic_rows": len(selected),
            "independent_scenarios": len(
                {row["scenario_id"] for row in selected}
            ),
            "metrics_vs_reactive": metrics,
        }
    if rows:
        scenario_rows = aggregate_scenario_relationship(rows)
        ade = np.asarray([row["ade_m"] for row in scenario_rows], dtype=float)
        gain = np.asarray(
            [row["delta_goodput_mbps"] for row in scenario_rows], dtype=float
        )
        if len(scenario_rows) > 1:
            correlation = stats.spearmanr(ade, gain)
            rho = float(correlation.statistic)
            p_value = float(correlation.pvalue)
        else:
            rho = float("nan")
            p_value = float("nan")
        summary["ade_vs_realized_goodput_gain"] = {
            "spearman_rho": rho,
            "p_value": p_value,
            "paired_model_traffic_rows": len(rows),
            "independent_scenarios": len(scenario_rows),
            "aggregation": (
                "ADE and goodput gain are averaged across model objectives/seeds "
                "and paired traffic realizations within each WOMD scenario before "
                "correlation."
            ),
            "interpretation": (
                "This scenario-level association is descriptive and does not by "
                "itself establish a causal scheduler gain."
            ),
        }
    return summary


def write_utility_analysis(
    rows: list[dict[str, Any]], output_dir: str | Path
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No aligned held-out/scheduler rows were found.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "ade_vs_scheduler_utility.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    scenario_csv = destination / "ade_vs_scheduler_utility_by_scenario.csv"
    scenario_rows = aggregate_scenario_relationship(rows)
    with scenario_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scenario_rows[0].keys())
        writer.writeheader()
        writer.writerows(scenario_rows)
    summary_path = destination / "scheduler_utility_summary.json"
    summary_path.write_text(
        json.dumps(summarize_utility(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "scenario_csv": scenario_csv,
        "summary": summary_path,
    }
