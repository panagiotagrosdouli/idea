"""Official held-out/OOD evaluation of all 20 frozen learned checkpoints."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import LinkConfig
from ..data.manifest import sha256_file
from ..learning.ablation import CANONICAL_SEEDS, OBJECTIVES, validate_training_resume
from ..learning.heldout import evaluate_checkpoint_arrays
from ..learning.inference import TorchCheckpointPredictor
from ..link import LinkModel
from .freeze import verify_publication_training_freeze

METRIC_NAMES = (
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


def _validate_official_npz(
    official_npz: Path,
    *,
    freeze: dict[str, object],
) -> tuple[str, dict[str, np.ndarray]]:
    if not official_npz.is_file():
        raise FileNotFoundError(f"official evaluation artifact not found: {official_npz}")
    with np.load(official_npz, allow_pickle=False) as data:
        split_values = np.asarray(data["split"]).astype(str)
        unique = set(split_values.tolist())
        if len(unique) != 1:
            raise ValueError("official evaluation NPZ must contain one split")
        split = unique.pop()
        if split not in {"held_out_test", "ood_test"}:
            raise ValueError("official evaluation NPZ is not held-out or OOD")
        training_hash = str(np.asarray(data["freeze_training_npz_sha256"]).item())
        dataset_hash = str(
            np.asarray(data["freeze_dataset_manifest_sha256"]).item()
        )
        completion_hash = str(
            np.asarray(data["freeze_completion_manifest_sha256"]).item()
        )
        if training_hash != freeze["training_npz_sha256"]:
            raise PermissionError("official NPZ training freeze hash mismatch")
        if dataset_hash != freeze["dataset_manifest_sha256"]:
            raise PermissionError("official NPZ dataset freeze hash mismatch")
        if completion_hash != freeze["completion_manifest_sha256"]:
            raise PermissionError("official NPZ completion freeze hash mismatch")
        arrays = {
            "history_xy": np.asarray(data["history_xy"]),
            "future_xy": np.asarray(data["future_xy"]),
            "future_ego_heading_rad": np.asarray(data["future_ego_heading_rad"]),
            "scenario_id": np.asarray(data["scenario_id"]),
        }
    return split, arrays


def _objective_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["objective"])].append(row)
    summary: dict[str, object] = {}
    for objective in OBJECTIVES:
        selected = grouped[objective]
        by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            by_scenario[str(row["scenario_id"])].append(row)
        scenario_means = []
        for scenario_id, scenario_rows in sorted(by_scenario.items()):
            seeds = {int(row["seed"]) for row in scenario_rows}
            if seeds != set(CANONICAL_SEEDS):
                raise RuntimeError(
                    f"{objective}/{scenario_id} does not contain all five seeds"
                )
            scenario_means.append(
                {
                    "scenario_id": scenario_id,
                    **{
                        metric: float(
                            np.mean([float(row[metric]) for row in scenario_rows])
                        )
                        for metric in METRIC_NAMES
                    },
                }
            )
        summary[objective] = {
            "scenario_count": len(scenario_means),
            "seed_count": len(CANONICAL_SEEDS),
            "scenario_seed_averaged_metrics": {
                metric: {
                    "mean": float(
                        np.mean([float(row[metric]) for row in scenario_means])
                    ),
                    "median": float(
                        np.median([float(row[metric]) for row in scenario_means])
                    ),
                    "std": float(
                        np.std(
                            [float(row[metric]) for row in scenario_means],
                            ddof=1,
                        )
                    )
                    if len(scenario_means) > 1
                    else 0.0,
                }
                for metric in METRIC_NAMES
            },
            "scenario_rows": scenario_means,
        }
    return summary


def run_official_learned_evaluation(
    dataset_dir: str | Path,
    *,
    training_npz: str | Path,
    ablation_dir: str | Path,
    official_npz: str | Path,
    output_path: str | Path,
    link_config: LinkConfig,
    batch_size: int = 1024,
    dt_s: float = 0.1,
) -> dict[str, object]:
    """Evaluate every objective/seed checkpoint on a frozen official split."""
    freeze = verify_publication_training_freeze(dataset_dir, ablation_dir, training_npz)
    split, arrays = _validate_official_npz(Path(official_npz), freeze=freeze)
    training_sha = str(freeze["training_npz_sha256"])
    link_model = LinkModel(link_config)
    root = Path(ablation_dir)
    rows: list[dict[str, object]] = []

    for objective in OBJECTIVES:
        for seed in CANONICAL_SEEDS:
            run_dir = root / objective / f"seed_{seed}"
            result_path = run_dir / "training_result.json"
            validation = validate_training_resume(
                result_path,
                expected_objective=objective,
                expected_seed=seed,
                expected_dataset_sha256=training_sha,
                expected_run_dir=run_dir,
            )
            if not validation.valid or validation.result is None:
                raise PermissionError(
                    f"unverified official checkpoint {objective}/{seed}: "
                    f"{validation.reason}"
                )
            predictor = TorchCheckpointPredictor(validation.result.checkpoint)
            checkpoint_rows = evaluate_checkpoint_arrays(
                predictor=predictor,
                history_xy=arrays["history_xy"],
                future_xy=arrays["future_xy"],
                future_ego_heading_rad=arrays["future_ego_heading_rad"],
                scenario_ids=arrays["scenario_id"],
                link_model=link_model,
                checkpoint=validation.result.checkpoint,
                objective=objective,
                seed=seed,
                batch_size=batch_size,
                dt_s=dt_s,
            )
            rows.extend(asdict(row) for row in checkpoint_rows)

    expected_pairs = len(OBJECTIVES) * len(CANONICAL_SEEDS)
    observed_pairs = {(str(row["objective"]), int(row["seed"])) for row in rows}
    if len(observed_pairs) != expected_pairs:
        raise RuntimeError("official learned evaluation is missing objective/seed pairs")

    report = {
        "status": "COMPLETED",
        "protocol": "synthetic_dataset_v1_official_learned_evaluation",
        "split": split,
        "official_npz": str(official_npz),
        "official_npz_sha256": sha256_file(official_npz),
        "freeze": freeze,
        "objectives": list(OBJECTIVES),
        "seeds": list(CANONICAL_SEEDS),
        "rows": rows,
        "objective_summary": _objective_summary(rows),
        "scientific_guards": {
            "all_20_checkpoints_evaluated": True,
            "scenario_level_summary": True,
            "five_training_seeds_averaged_within_scenario_for_objective_summary": True,
            "no_official_split_model_selection": True,
            "negative_results_preserved": True,
        },
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite official learned evaluation: {destination}"
        )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
