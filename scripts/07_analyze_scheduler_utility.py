from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictive_pc_fmcw.learning.utility_analysis import (
    aggregate_scenario_relationship,
    join_accuracy_and_scheduler_utility,
    write_utility_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test ADE versus realized packet-scheduler communication utility."
    )
    parser.add_argument("heldout_csv")
    parser.add_argument("scheduler_root")
    parser.add_argument("--output", default="artifacts/ade_vs_scheduler_utility")
    args = parser.parse_args()
    destination = Path(args.output)
    rows = join_accuracy_and_scheduler_utility(
        args.heldout_csv, args.scheduler_root
    )
    artifacts = write_utility_analysis(rows, destination)
    scenario_rows = aggregate_scenario_relationship(rows)

    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    axis.scatter(
        [row["ade_m"] for row in scenario_rows],
        [row["delta_goodput_mbps"] for row in scenario_rows],
        s=24,
        alpha=0.65,
    )
    axis.axhline(0, linewidth=0.8, linestyle="--")
    axis.set_xlabel("Scenario-mean held-out ADE (m)")
    axis.set_ylabel("Scenario-mean learned minus reactive goodput (Mbps)")
    axis.set_title("Trajectory accuracy vs realized communication utility")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(
            destination / f"ade_vs_realized_goodput.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    print({key: str(value) for key, value in artifacts.items()})


if __name__ == "__main__":
    main()
