"""Frozen scheduler-family mapping for the dataset-free publication protocol."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulerFamily:
    protocol_id: str
    scheduler_name: str
    forecast_source: str
    learned_objective: str | None = None
    deployable: bool = True


SCHEDULER_FAMILIES = (
    SchedulerFamily("S0", "reactive_greedy", "current_link"),
    SchedulerFamily("S1", "cv_predictive", "constant_velocity"),
    SchedulerFamily(
        "S2",
        "learned_predictive",
        "trajectory_gru",
        learned_objective="trajectory_only",
    ),
    SchedulerFamily(
        "S3",
        "learned_predictive",
        "communication_aware_gru",
        learned_objective="full_communication_aware",
    ),
    SchedulerFamily("S4", "oracle", "oracle_future", deployable=False),
)


def validate_scheduler_protocol() -> None:
    """Fail closed if the publication scheduler family drifts."""
    if tuple(item.protocol_id for item in SCHEDULER_FAMILIES) != (
        "S0",
        "S1",
        "S2",
        "S3",
        "S4",
    ):
        raise ValueError("publication scheduler IDs must remain S0-S4")
    if SCHEDULER_FAMILIES[-1].deployable:
        raise ValueError("oracle scheduler must remain evaluator-only")
    learned = [item for item in SCHEDULER_FAMILIES if item.learned_objective]
    if len(learned) != 2:
        raise ValueError("publication protocol requires exactly two learned schedulers")
    if {item.learned_objective for item in learned} != {
        "trajectory_only",
        "full_communication_aware",
    }:
        raise ValueError("learned scheduler objectives drifted from frozen protocol")


def scheduler_protocol_manifest() -> dict[str, object]:
    validate_scheduler_protocol()
    return {
        "paired_inputs": (
            "mobility_scenario",
            "channel_realization",
            "packet_arrivals",
            "packet_deadlines",
            "traffic_seed",
        ),
        "inferential_unit": "scenario_episode",
        "families": [
            {
                "protocol_id": item.protocol_id,
                "scheduler_name": item.scheduler_name,
                "forecast_source": item.forecast_source,
                "learned_objective": item.learned_objective,
                "deployable": item.deployable,
            }
            for item in SCHEDULER_FAMILIES
        ],
    }
