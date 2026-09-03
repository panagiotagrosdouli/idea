from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")


METRICS = (
    "validation_ade_m",
    "validation_fde_m",
    "validation_trajectory_loss",
    "validation_link_loss",
    "validation_outage_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize complete or partial learned-training ablations."
    )
    parser.add_argument("ablation_dir")
    parser.add_argument("--output", default="artifacts/training_ablation_analysis")
    args = parser.parse_args()
    source = Path(args.ablation_dir)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    result_paths = sorted(source.glob("*/seed_*/training_result.json"))
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    if not rows:
        raise ValueError("No completed training_result.json files were found.")
    dataset_hashes = {row["dataset_sha256"] for row in rows}
    if len(dataset_hashes) != 1:
        raise ValueError("Ablation results do not share one dataset SHA-256.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["objective"]].append(row)
    summary = {
        "status": "complete" if len(rows) == 12 else "partial",
        "completed_runs": len(rows),
        "expected_runs": 12,
        "dataset_sha256": next(iter(dataset_hashes)),
        "scope_warning": (
            "Internal development metrics are training diagnostics, not official "
            "WOMD held-out or packet-scheduler evidence."
        ),
        "objectives": {},
    }
    for objective, selected in sorted(grouped.items()):
        summary["objectives"][objective] = {
            "runs": len(selected),
            "seeds": sorted(int(row["seed"]) for row in selected),
            **{
                metric: {
                    "mean": float(np.mean([row[metric] for row in selected])),
                    "std": float(np.std([row[metric] for row in selected], ddof=1))
                    if len(selected) > 1
                    else None,
                }
                for metric in METRICS
            },
        }
    paired = {}
    baseline = {row["seed"]: row for row in grouped.get("trajectory_only", [])}
    for objective, selected in sorted(grouped.items()):
        if objective == "trajectory_only":
            continue
        current = {row["seed"]: row for row in selected}
        seeds = sorted(set(baseline) & set(current))
        if len(seeds) < 2:
            continue
        paired[objective] = {}
        for metric in METRICS:
            differences = np.asarray(
                [current[seed][metric] - baseline[seed][metric] for seed in seeds]
            )
            paired[objective][metric] = {
                "paired_seeds": seeds,
                "mean_difference_vs_trajectory_only": float(differences.mean()),
                "paired_t_p_value": float(stats.ttest_1samp(differences, 0).pvalue),
                "wilcoxon_p_value": float(stats.wilcoxon(differences).pvalue),
            }
    summary["paired_vs_trajectory_only"] = paired
    (destination / "training_ablation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (destination / "training_ablation_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    objectives = sorted(summary["objectives"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for axis, metric, label in (
        (axes[0], "validation_ade_m", "Internal development ADE (m)"),
        (axes[1], "validation_link_loss", "Internal development link loss"),
    ):
        means = [
            summary["objectives"][name][metric]["mean"] for name in objectives
        ]
        errors = [
            summary["objectives"][name][metric]["std"] or 0
            for name in objectives
        ]
        axis.bar(objectives, means, yerr=errors, capsize=4, color="#2563eb")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(f"Learned ablation diagnostics ({len(rows)}/12 completed runs)")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            destination / f"training_ablation_partial.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
