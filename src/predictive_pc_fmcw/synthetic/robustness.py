"""Frozen robustness sweeps for the dataset-free publication study."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from ..config import ExperimentConfig, LinkConfig, SensingConfig
from ..data.manifest import sha256_file
from .scheduling_evaluation import run_synthetic_scheduling_evaluation


@dataclass(frozen=True)
class RobustnessCondition:
    name: str
    sensing_range_multiplier: float = 1.0
    sensing_bearing_multiplier: float = 1.0
    forecast_channel_mode: str | None = None
    forecast_reference_snr_offset_db: float = 0.0
    forecast_attenuation_multiplier: float = 1.0


ROBUSTNESS_CONDITIONS = (
    RobustnessCondition("nominal"),
    RobustnessCondition(
        "observation_noise_x2",
        sensing_range_multiplier=2.0,
        sensing_bearing_multiplier=2.0,
    ),
    RobustnessCondition(
        "observation_noise_x4",
        sensing_range_multiplier=4.0,
        sensing_bearing_multiplier=4.0,
    ),
    RobustnessCondition(
        "forecast_range_pointing_only",
        forecast_channel_mode="range_pointing",
    ),
    RobustnessCondition(
        "forecast_range_only",
        forecast_channel_mode="range_only",
    ),
    RobustnessCondition(
        "forecast_snr_minus_3db",
        forecast_reference_snr_offset_db=-3.0,
    ),
    RobustnessCondition(
        "forecast_snr_plus_3db",
        forecast_reference_snr_offset_db=3.0,
    ),
    RobustnessCondition(
        "forecast_attenuation_x1_5",
        forecast_attenuation_multiplier=1.5,
    ),
)


def validate_robustness_protocol() -> None:
    names = tuple(condition.name for condition in ROBUSTNESS_CONDITIONS)
    if len(names) != len(set(names)):
        raise ValueError("robustness condition names must be unique")
    if names[0] != "nominal":
        raise ValueError("robustness protocol must begin with the nominal condition")
    for condition in ROBUSTNESS_CONDITIONS:
        if condition.sensing_range_multiplier <= 0:
            raise ValueError("range-noise multipliers must be positive")
        if condition.sensing_bearing_multiplier <= 0:
            raise ValueError("bearing-noise multipliers must be positive")
        if condition.forecast_attenuation_multiplier <= 0:
            raise ValueError("attenuation multipliers must be positive")
        if condition.forecast_channel_mode not in {
            None,
            "range_only",
            "range_pointing",
            "full",
        }:
            raise ValueError("unsupported forecast channel mode")


def robustness_protocol_manifest() -> dict[str, object]:
    validate_robustness_protocol()
    return {
        "conditions": [asdict(condition) for condition in ROBUSTNESS_CONDITIONS],
        "actual_channel_frozen_across_conditions": True,
        "same_episodes_and_paired_traffic_seeds_across_conditions": True,
        "ood_evaluated_separately": True,
        "radial_velocity_scheduler_limitation_declared": True,
    }


def _condition_sensing(
    nominal: SensingConfig,
    condition: RobustnessCondition,
) -> SensingConfig:
    return replace(
        nominal,
        range_std_base_m=(
            nominal.range_std_base_m * condition.sensing_range_multiplier
        ),
        range_std_per_m=(
            nominal.range_std_per_m * condition.sensing_range_multiplier
        ),
        bearing_std_deg=(
            nominal.bearing_std_deg * condition.sensing_bearing_multiplier
        ),
        assumption_source=f"robustness condition {condition.name}",
    )


def _condition_forecast_link(
    actual: LinkConfig,
    condition: RobustnessCondition,
) -> LinkConfig:
    return replace(
        actual,
        channel_mode=condition.forecast_channel_mode or actual.channel_mode,
        reference_snr_db=(
            actual.reference_snr_db + condition.forecast_reference_snr_offset_db
        ),
        atmospheric_attenuation_per_m=(
            actual.atmospheric_attenuation_per_m
            * condition.forecast_attenuation_multiplier
        ),
    )


def run_synthetic_robustness_sweep(
    dataset_dir: str | Path,
    *,
    split: str,
    ablation_dir: str | Path,
    training_npz: str | Path,
    selection_manifest: str | Path,
    config: ExperimentConfig,
    output_dir: str | Path,
    vehicles_per_episode: int = 5,
    history_steps: int = 20,
) -> dict[str, object]:
    """Run every frozen robustness condition without changing actual physics."""
    validate_robustness_protocol()
    destination = Path(output_dir)
    manifest_path = destination / "robustness_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite robustness sweep: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=True)

    dataset_manifest = json.loads(
        (Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    observation = dataset_manifest["observation_config"]
    if not isinstance(observation, dict):
        raise ValueError("dataset observation_config is invalid")
    nominal_sensing = SensingConfig(
        model="range_bearing_assumed",
        range_std_base_m=float(observation["range_std_m"]),
        range_std_per_m=0.0,
        bearing_std_deg=float(observation["bearing_std_rad"]) * 180.0 / 3.141592653589793,
        temporal_correlation=0.0,
        covariance_aware=True,
        assumption_source="synthetic_dataset_v1 frozen observation configuration",
    )

    artifacts: list[dict[str, object]] = []
    expected_plan: dict[str, object] | None = None
    for condition in ROBUSTNESS_CONDITIONS:
        output_path = destination / f"{condition.name}.json"
        report = run_synthetic_scheduling_evaluation(
            dataset_dir,
            split=split,
            ablation_dir=ablation_dir,
            training_npz=training_npz,
            selection_manifest=selection_manifest,
            config=config,
            output_path=output_path,
            vehicles_per_episode=vehicles_per_episode,
            history_steps=history_steps,
            sensing_override=_condition_sensing(nominal_sensing, condition),
            forecast_link_config=_condition_forecast_link(config.link, condition),
            condition_label=condition.name,
        )
        plan = report["plan"]
        if not isinstance(plan, dict):
            raise RuntimeError("robustness condition returned an invalid plan")
        if expected_plan is None:
            expected_plan = plan
        elif plan != expected_plan:
            raise RuntimeError("robustness conditions did not use identical run plans")
        artifacts.append(
            {
                "condition": condition.name,
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "planned_runs": plan["planned_runs"],
            }
        )

    manifest = {
        "status": "COMPLETED",
        "protocol": robustness_protocol_manifest(),
        "split": split,
        "plan": expected_plan,
        "artifacts": artifacts,
        "scientific_guards": {
            "actual_channel_unchanged": True,
            "forecast_mismatch_explicit": True,
            "same_episode_composition": True,
            "same_paired_traffic_seed_set": True,
            "negative_results_preserved": True,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
