from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from predictive_pc_fmcw.learning.analysis import (
    load_heldout_rows,
    summarize_objectives,
    write_learned_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create held-out learned-model tables, statistics and figures."
    )
    parser.add_argument("heldout_csv")
    parser.add_argument("--output", default="artifacts/learned_paper_analysis")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    rows = load_heldout_rows(args.heldout_csv)
    report_path = write_learned_analysis(rows, destination)
    summary = summarize_objectives(rows)

    objectives = sorted(summary)
    columns = ["ade_m", "fde_m", "snr_mae_db", "goodput_mae_mbps", "outage_f1"]
    header = "Objective & ADE (m) & FDE (m) & SNR MAE (dB) & Goodput MAE (Mbps) & Outage F1 \\\\"
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule", header, "\\midrule"]
    for objective in objectives:
        values = [summary[objective][column]["mean"] for column in columns]
        lines.append(
            objective.replace("_", "\\_") + " & "
            + " & ".join(f"{value:.3f}" for value in values)
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (destination / "learned_heldout_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    colors = {name: color for name, color in zip(
        objectives, ["#2563eb", "#dc2626", "#0f766e", "#7c3aed"], strict=False
    )}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    for objective in objectives:
        selected = [row for row in rows if row["objective"] == objective]
        axes[0].scatter(
            [row["ade_m"] for row in selected],
            [row["snr_mae_db"] for row in selected],
            s=13, alpha=0.35, color=colors[objective], label=objective,
        )
        axes[1].scatter(
            [row["ade_m"] for row in selected],
            [row["goodput_mae_mbps"] for row in selected],
            s=13, alpha=0.35, color=colors[objective], label=objective,
        )
    axes[0].set(xlabel="ADE (m)", ylabel="Future-SNR MAE (dB)")
    axes[1].set(xlabel="ADE (m)", ylabel="Future-goodput MAE (Mbps)")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Trajectory accuracy versus held-out link-state fidelity")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"ade_vs_link_fidelity.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"analysis": str(report_path), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
