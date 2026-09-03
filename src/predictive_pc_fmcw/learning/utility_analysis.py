from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from ..metrics import paired_metric_statistics


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
        seed = int(metrics_path.parent.name.removeprefix("seed_"))
        rows = _read_csv(metrics_path)
        indexed = {(row["scheduler"], row["scenario_id"]): row for row in rows}
        scenarios = sorted(
            scenario
            for scheduler, scenario in indexed
            if scheduler == "learned_predictive"
            and ("reactive_greedy", scenario) in indexed
            and (objective, seed, scenario) in accuracy
        )
        for scenario in scenarios:
            learned = indexed[("learned_predictive", scenario)]
            reactive = indexed[("reactive_greedy", scenario)]
            motion = accuracy[(objective, seed, scenario)]
            item: dict[str, Any] = {
                "objective": objective,
                "seed": seed,
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
        summary[objective] = {
            "scenario_seed_rows": len(selected),
            "independent_scenarios": len(
                {row["scenario_id"] for row in selected}
            ),
            "metrics_vs_reactive": metrics,
        }
    if rows:
        ade = np.asarray([row["ade_m"] for row in rows], dtype=float)
        gain = np.asarray([row["delta_goodput_mbps"] for row in rows], dtype=float)
        correlation = stats.spearmanr(ade, gain)
        summary["ade_vs_realized_goodput_gain"] = {
            "spearman_rho": float(correlation.statistic),
            "p_value": float(correlation.pvalue),
            "rows": len(rows),
            "interpretation": (
                "A non-monotonic or weak relationship supports non-equivalence; "
                "it does not by itself prove a causal scheduler gain."
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
    summary_path = destination / "scheduler_utility_summary.json"
    summary_path.write_text(
        json.dumps(summarize_utility(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary": summary_path}
