from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.learning.calibration import fit_residual_gaussian
from predictive_pc_fmcw.learning.completion import verify_completion_manifest
from predictive_pc_fmcw.learning.heldout import evaluate_checkpoint_arrays
from predictive_pc_fmcw.learning.inference import TorchCheckpointPredictor
from predictive_pc_fmcw.link import LinkModel
from predictive_pc_fmcw.link_verification import verify_lut


def _predict_in_batches(predictor, history, horizon, batch_size):
    chunks = []
    for start in range(0, len(history), batch_size):
        chunks.append(
            predictor.predict(history[start : start + batch_size], horizon, 0.1)
        )
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate learned checkpoints on untouched official WOMD validation."
        )
    )
    parser.add_argument("validation_npz")
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--output", default="artifacts/heldout_learned_evaluation")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device")
    parser.add_argument(
        "--development-npz",
        help="Development-only NPZ used to fit residual Gaussian uncertainty.",
    )
    parser.add_argument(
        "--completion-manifest",
        help="Stage-4 manifest proving the frozen 20-checkpoint ablation is complete.",
    )
    parser.add_argument(
        "--ber-lut",
        help="Verified Stage-2 Part-A LUT used for held-out link metrics.",
    )
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)

    with np.load(args.validation_npz, allow_pickle=False) as data:
        split_labels = set(np.asarray(data["split"]).astype(str).tolist())
        if split_labels != {"official_validation"}:
            raise ValueError(
                "Held-out evaluation requires split=official_validation for every "
                f"sample; received {sorted(split_labels)}."
            )
        validation_arrays = {
            "history_xy": np.asarray(data["history_xy"]),
            "future_xy": np.asarray(data["future_xy"]),
            "future_ego_heading_rad": np.asarray(data["future_ego_heading_rad"]),
            "scenario_id": np.asarray(data["scenario_id"]),
        }

    completion_report = None
    if args.completion_manifest:
        if not args.development_npz:
            raise ValueError(
                "completion-manifest verification requires --development-npz."
            )
        completion_report = verify_completion_manifest(
            args.completion_manifest,
            training_npz=args.development_npz,
            checkpoints=list(args.checkpoints),
        )
        if completion_report["status"] != "PASS":
            raise ValueError("Stage-4 completion manifest verification failed.")

    config = load_config(args.config)
    link_config = config.link
    lut_report = None
    if args.ber_lut:
        lut_report = verify_lut(args.ber_lut)
        if lut_report["status"] != "PASS":
            raise ValueError("The supplied BER LUT does not satisfy the Stage-2 gate.")
        link_config = replace(
            link_config,
            ber_source="lut",
            ber_lut_path=str(Path(args.ber_lut)),
        )
    link_model = LinkModel(link_config)

    all_rows = []
    development = None
    development_mask = None
    if args.development_npz:
        with np.load(args.development_npz, allow_pickle=False) as source:
            development = {
                "history_xy": np.asarray(source["history_xy"]),
                "future_xy": np.asarray(source["future_xy"]),
                "split": np.asarray(source["split"]).astype(str),
            }
        development_mask = development["split"] == "development"
        if not np.any(development_mask):
            raise ValueError(
                "Uncertainty calibration requires at least one development sample."
            )

    for checkpoint in args.checkpoints:
        predictor = TorchCheckpointPredictor(checkpoint, device=args.device)
        payload = predictor._torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        training = payload.get("training", {})
        calibration = None
        if development is not None and development_mask is not None:
            development_history = development["history_xy"][development_mask]
            development_future = development["future_xy"][development_mask]
            predicted_development = _predict_in_batches(
                predictor,
                development_history,
                development_future.shape[1],
                args.batch_size,
            )
            calibration = fit_residual_gaussian(
                predicted_development, development_future
            )
            calibration_path = destination / (
                f"calibration_{training.get('objective', 'unknown')}_"
                f"seed_{training.get('seed', -1)}.json"
            )
            calibration_path.write_text(
                json.dumps(calibration.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        all_rows.extend(
            evaluate_checkpoint_arrays(
                predictor=predictor,
                history_xy=validation_arrays["history_xy"],
                future_xy=validation_arrays["future_xy"],
                future_ego_heading_rad=validation_arrays[
                    "future_ego_heading_rad"
                ],
                scenario_ids=validation_arrays["scenario_id"],
                link_model=link_model,
                checkpoint=str(checkpoint),
                objective=str(training.get("objective", "unknown")),
                seed=int(training.get("seed", -1)),
                batch_size=args.batch_size,
                calibration=calibration,
            )
        )

    dictionaries = [asdict(row) for row in all_rows]
    if not dictionaries:
        raise ValueError("Held-out evaluation produced no scenario rows.")
    csv_path = destination / "heldout_metrics_by_scenario.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dictionaries[0].keys())
        writer.writeheader()
        writer.writerows(dictionaries)
    summary = {}
    metrics = [
        "ade_m",
        "fde_m",
        "range_mae_m",
        "bearing_mae_deg",
        "snr_mae_db",
        "goodput_mae_mbps",
        "outage_f1",
        "outage_auroc",
        "link_lifetime_mae_s",
        "gaussian_nll",
        "coverage_50",
        "coverage_90",
        "coverage_95",
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
    provenance = {
        "validation_npz": args.validation_npz,
        "development_npz": args.development_npz,
        "completion_verification": completion_report,
        "ber_lut_verification": lut_report,
        "link_model": {
            "ber_source": link_config.ber_source,
            "ber_lut_path": link_config.ber_lut_path,
            "received_power_calibrated": link_config.received_power_calibrated,
        },
    }
    (destination / "heldout_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
