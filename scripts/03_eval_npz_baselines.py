from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.learning.heldout import evaluate_checkpoint_arrays
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    InteractingMultipleModelPredictor,
    KalmanConstantVelocityPredictor,
    LastPositionPredictor,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate causal classical baselines directly on a WOMD NPZ."
    )
    parser.add_argument("npz")
    parser.add_argument("--split", default="development")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output", default="artifacts/motion_baselines_npz")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    archive = np.load(args.npz, allow_pickle=False)
    labels = np.asarray(archive["split"]).astype(str)
    selected = labels == args.split
    if not np.any(selected):
        raise ValueError(f"No samples with split={args.split!r}.")
    predictors = (
        LastPositionPredictor(),
        ConstantVelocityPredictor(),
        ConstantAccelerationPredictor(),
        KalmanConstantVelocityPredictor(),
        InteractingMultipleModelPredictor(),
    )
    link_model = LinkModel(load_config(args.config).link)
    rows = []
    for predictor in predictors:
        rows.extend(
            evaluate_checkpoint_arrays(
                predictor=predictor,
                history_xy=archive["history_xy"][selected],
                future_xy=archive["future_xy"][selected],
                future_ego_heading_rad=archive["future_ego_heading_rad"][selected],
                scenario_ids=archive["scenario_id"][selected],
                link_model=link_model,
                checkpoint="classical",
                objective=predictor.name,
                seed=-1,
                batch_size=args.batch_size,
            )
        )
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    dictionaries = [asdict(row) for row in rows]
    with (destination / "forecast_metrics_by_scenario.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=dictionaries[0].keys())
        writer.writeheader()
        writer.writerows(dictionaries)
    metrics = ("ade_m", "fde_m", "snr_mae_db", "outage_f1", "link_lifetime_mae_s")
    summary = {
        predictor.name: {
            metric: float(
                np.nanmean(
                    [
                        getattr(row, metric)
                        for row in rows
                        if row.objective == predictor.name
                    ]
                )
            )
            for metric in metrics
        }
        for predictor in predictors
    }
    (destination / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
