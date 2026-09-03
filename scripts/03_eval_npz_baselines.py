from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.learning.heldout import evaluate_checkpoint_arrays
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.link_verification import verify_lut
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
    parser.add_argument(
        "--ber-lut",
        help="Verified Stage-2 Part-A LUT. Required for canonical paper runs.",
    )
    parser.add_argument("--output", default="artifacts/motion_baselines_npz")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as archive:
        labels = np.asarray(archive["split"]).astype(str)
        selected = labels == args.split
        if not np.any(selected):
            raise ValueError(f"No samples with split={args.split!r}.")
        history_xy = archive["history_xy"][selected]
        future_xy = archive["future_xy"][selected]
        future_ego_heading_rad = archive["future_ego_heading_rad"][selected]
        scenario_ids = archive["scenario_id"][selected]

    predictors = (
        LastPositionPredictor(),
        ConstantVelocityPredictor(),
        ConstantAccelerationPredictor(),
        KalmanConstantVelocityPredictor(),
        InteractingMultipleModelPredictor(),
    )
    config = load_config(args.config)
    link_config = config.link
    if args.ber_lut:
        verification = verify_lut(args.ber_lut)
        if verification["status"] != "PASS":
            raise ValueError("The supplied BER LUT does not satisfy the Stage-2 gate.")
        link_config = replace(
            link_config,
            ber_source="lut",
            ber_lut_path=str(Path(args.ber_lut)),
        )
    link_model = LinkModel(link_config)

    rows = []
    for predictor in predictors:
        rows.extend(
            evaluate_checkpoint_arrays(
                predictor=predictor,
                history_xy=history_xy,
                future_xy=future_xy,
                future_ego_heading_rad=future_ego_heading_rad,
                scenario_ids=scenario_ids,
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
    summary["link_model"] = {
        "ber_source": link_config.ber_source,
        "ber_lut_path": link_config.ber_lut_path,
        "received_power_calibrated": link_config.received_power_calibrated,
    }
    (destination / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
