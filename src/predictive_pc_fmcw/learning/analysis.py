from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import holm_adjusted_pvalues, paired_metric_statistics

METRICS = (
    "ade_m",
    "fde_m",
    "range_mae_m",
    "bearing_mae_deg",
    "snr_mae_db",
    "goodput_mae_mbps",
    "outage_f1",
    "outage_auroc",
    "link_lifetime_mae_s",
)


def load_heldout_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Held-out metrics CSV is empty.")
    required = {"objective", "seed", "scenario_id", *METRICS}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Held-out metrics CSV is missing columns: {sorted(missing)}")
    converted = []
    for row in rows:
        item = dict(row)
        item["seed"] = int(row["seed"])
        for metric in METRICS:
            item[metric] = float(row[metric])
        converted.append(item)
    return converted


def summarize_objectives(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["objective"])].append(row)
    summary = {}
    for objective, selected in sorted(grouped.items()):
        summary[objective] = {
            "scenario_seed_rows": len(selected),
            "scenarios": len({row["scenario_id"] for row in selected}),
            "seeds": len({row["seed"] for row in selected}),
            **{
                metric: {
                    "mean": float(np.nanmean([row[metric] for row in selected])),
                    "std": float(np.nanstd([row[metric] for row in selected], ddof=1))
                    if len(selected) > 1
                    else 0.0,
                }
                for metric in METRICS
            },
        }
    return summary


def _apply_holm_family(results: dict[str, dict[str, Any]]) -> None:
    names = list(results)
    t_adjusted = holm_adjusted_pvalues(
        results[name]["paired_t_test_p_value"] for name in names
    )
    wilcoxon_adjusted = holm_adjusted_pvalues(
        results[name]["wilcoxon_p_value"] for name in names
    )
    for name, t_value, wilcoxon_value in zip(
        names, t_adjusted, wilcoxon_adjusted, strict=True
    ):
        results[name]["paired_t_test_holm_p_value"] = t_value
        results[name]["wilcoxon_holm_p_value"] = wilcoxon_value


def paired_full_vs_trajectory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {
        (str(row["objective"]), int(row["seed"]), str(row["scenario_id"])): row
        for row in rows
    }
    keys = sorted(
        {
            (seed, scenario)
            for objective, seed, scenario in indexed
            if objective == "full" and ("trajectory_only", seed, scenario) in indexed
        }
    )
    if not keys:
        return {"pairs": 0, "metrics": {}}
    results = {}
    for metric in METRICS:
        proposed = np.asarray([indexed[("full", *key)][metric] for key in keys])
        baseline = np.asarray(
            [indexed[("trajectory_only", *key)][metric] for key in keys]
        )
        results[metric] = paired_metric_statistics(
            proposed,
            baseline,
            higher_is_better=metric in {"outage_f1", "outage_auroc"},
            clusters=[scenario for _, scenario in keys],
        )
    _apply_holm_family(results)
    return {"pairs": len(keys), "metrics": results}


def aggregate_scenario_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average dependent model-seed rows within each independent WOMD scenario."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scenario_id"])].append(row)
    return [
        {
            "scenario_id": scenario_id,
            **{
                metric: float(np.nanmean([row[metric] for row in selected]))
                for metric in METRICS
            },
            "model_seed_rows": len(selected),
        }
        for scenario_id, selected in sorted(grouped.items())
    ]


def accuracy_link_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    scenario_rows = aggregate_scenario_metrics(rows)
    ade = np.asarray([row["ade_m"] for row in scenario_rows], dtype=np.float64)
    results: dict[str, Any] = {
        "model_seed_rows": len(rows),
        "independent_scenarios": len(scenario_rows),
        "aggregation": (
            "Metrics are averaged across objectives/model seeds within each WOMD "
            "scenario before Spearman correlation."
        ),
    }
    for metric in ("snr_mae_db", "goodput_mae_mbps", "link_lifetime_mae_s"):
        values = np.asarray([row[metric] for row in scenario_rows], dtype=np.float64)
        finite = np.isfinite(ade) & np.isfinite(values)
        if int(finite.sum()) > 1:
            statistic = spearmanr(ade[finite], values[finite])
            rho = float(statistic.statistic)
            p_value = float(statistic.pvalue)
        else:
            rho = float("nan")
            p_value = float("nan")
        results[f"ade_vs_{metric}_spearman_rho"] = rho
        results[f"ade_vs_{metric}_p_value"] = p_value
    return results


def write_learned_analysis(rows: list[dict[str, Any]], output: str | Path) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope_warning": (
            "Goodput MAE measures link-state prediction fidelity, not realized "
            "packet-scheduler utility. Scheduler-goodput claims require "
            "packet-level runs."
        ),
        "objective_summary": summarize_objectives(rows),
        "full_vs_trajectory_only": paired_full_vs_trajectory(rows),
        "accuracy_link_correlations": accuracy_link_correlations(rows),
    }
    path = destination / "learned_heldout_analysis.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path
