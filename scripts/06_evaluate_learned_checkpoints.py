from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.learning.heldout import evaluate_checkpoint_arrays
from predictive_pc_fmcw.learning.inference import TorchCheckpointPredictor
from predictive_pc_fmcw.link import LinkModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned checkpoints on untouched official WOMD validation."
    )
    parser.add_argument("validation_npz")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", default="artifacts/heldout_learned_evaluation")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device")
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    data = np.load(args.validation_npz, allow_pickle=False)
    split_labels = set(np.asarray(data["split"]).astype(str).tolist())
    if split_labels != {"official_validation"}:
        raise ValueError(
            "Held-out evaluation requires split=official_validation for every sample; "
            f"received {sorted(split_labels)}."
        )
    link_model = LinkModel(load_config(args.config).link)
    all_rows = []
    for checkpoint in args.checkpoints:
        predictor = TorchCheckpointPredictor(checkpoint, device=args.device)
        payload = predictor._torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        training = payload.get("training", {})
        all_rows.extend(
            evaluate_checkpoint_arrays(
                predictor=predictor,
                history_xy=data["history_xy"],
                future_xy=data["future_xy"],
                future_ego_heading_rad=data["future_ego_heading_rad"],
                scenario_ids=data["scenario_id"],
                link_model=link_model,
                checkpoint=str(checkpoint),
                objective=str(training.get("objective", "unknown")),
                seed=int(training.get("seed", -1)),
                batch_size=args.batch_size,
            )
        )
    dictionaries = [asdict(row) for row in all_rows]
    csv_path = destination / "heldout_metrics_by_scenario.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dictionaries[0].keys())
        writer.writeheader()
        writer.writerows(dictionaries)
    summary = {}
    metrics = [
        "ade_m", "fde_m", "range_mae_m", "bearing_mae_deg", "snr_mae_db",
        "goodput_mae_mbps", "outage_f1", "outage_auroc",
        "link_lifetime_mae_s",
    ]
    for objective in sorted({row.objective for row in all_rows}):
        chosen = [row for row in all_rows if row.objective == objective]
        summary[objective] = {
            "scenario_rows": len(chosen),
            **{
                metric: float(np.nanmean([getattr(row, metric) for row in chosen]))
                for metric in metrics
            },
        }
    (destination / "heldout_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
