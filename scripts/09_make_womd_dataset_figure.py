from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic paper figure for a WOMD training NPZ."
    )
    parser.add_argument("dataset")
    parser.add_argument("--output", default="artifacts/womd_official_dataset")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)

    with np.load(args.dataset, allow_pickle=False) as archive:
        history = archive["history_xy"]
        future = archive["future_xy"]
        split = archive["split"]
        current = history[:, -1]
        displacement = np.linalg.norm(future[:, -1] - current, axis=1)
        order = np.argsort(displacement)
        indices = order[np.linspace(0, len(order) - 1, 36, dtype=int)]

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        for index in indices:
            origin = history[index, -1]
            past = history[index] - origin
            prediction = future[index] - origin
            axes[0].plot(past[:, 0], past[:, 1], color="#2563eb", alpha=0.28)
            axes[0].plot(
                prediction[:, 0], prediction[:, 1], color="#dc2626", alpha=0.22
            )
        axes[0].scatter([0], [0], color="black", s=18, zorder=5)
        axes[0].set_title("True-SDC trajectory samples")
        axes[0].set_xlabel("Relative x (m)")
        axes[0].set_ylabel("Relative y (m)")
        axes[0].axis("equal")
        axes[0].grid(alpha=0.2)
        axes[0].plot([], [], color="#2563eb", label="1.1 s history")
        axes[0].plot([], [], color="#dc2626", label="8.0 s future")
        axes[0].legend(frameon=False, loc="best")

        axes[1].hist(
            displacement,
            bins=np.linspace(0, float(np.percentile(displacement, 99)), 50),
            color="#0f766e",
            edgecolor="white",
            linewidth=0.25,
        )
        axes[1].axvline(
            float(np.median(displacement)),
            color="#f59e0b",
            linestyle="--",
            label=f"median = {np.median(displacement):.1f} m",
        )
        axes[1].axvline(
            float(np.percentile(displacement, 95)),
            color="#7c3aed",
            linestyle=":",
            label=f"P95 = {np.percentile(displacement, 95):.1f} m",
        )
        axes[1].set_title("Eight-second displacement")
        axes[1].set_xlabel("Displacement (m)")
        axes[1].set_ylabel("Samples")
        axes[1].grid(axis="y", alpha=0.2)
        axes[1].legend(frameon=False)
        fig.suptitle(
            f"Official WOMD true-SDC training corpus: {len(history):,} samples; "
            f"{np.unique(split).size} internal splits",
            fontsize=11,
        )
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(
                destination / f"womd_dataset_profile.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


if __name__ == "__main__":
    main()
