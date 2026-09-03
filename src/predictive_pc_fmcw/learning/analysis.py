from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..metrics import paired_metric_statistics

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
    return {"pairs": len(keys), "metrics": results}


def accuracy_link_correlations(rows: list[dict[str, Any]]) -> dict[str, float]:
    from scipy.stats import spearmanr

    ade = np.asarray([row["ade_m"] for row in rows], dtype=np.float64)
    results = {}
    for metric in ("snr_mae_db", "goodput_mae_mbps", "link_lifetime_mae_s"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        finite = np.isfinite(ade) & np.isfinite(values)
        statistic = spearmanr(ade[finite], values[finite])
        results[f"ade_vs_{metric}_spearman_rho"] = float(statistic.statistic)
        results[f"ade_vs_{metric}_p_value"] = float(statistic.pvalue)
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
