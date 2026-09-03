from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    """Weights for the packet-level RL objective.

    Positive reward is attached to useful delivered information.  Penalties
    reflect reliability, deadline, outage and switching costs.  The default
    values are deliberately moderate and must be tuned only on development
    scenarios before official held-out evaluation.
    """

    delivered_mbit_weight: float = 1.0
    deadline_drop_weight: float = 1.0
    failed_attempt_weight: float = 0.10
    scheduled_outage_weight: float = 0.25
    switch_weight: float = 0.01
    fairness_gain_weight: float = 0.20


@dataclass(frozen=True)
class TransitionOutcome:
    delivered_bits: int
    deadline_drops: int
    failed_attempts: int
    scheduled_outage: bool
    switched_vehicle: bool
    fairness_before: float
    fairness_after: float

    def __post_init__(self) -> None:
        if self.delivered_bits < 0:
            raise ValueError("delivered_bits cannot be negative")
        if self.deadline_drops < 0:
            raise ValueError("deadline_drops cannot be negative")
        if self.failed_attempts < 0:
            raise ValueError("failed_attempts cannot be negative")


def compute_reward(outcome: TransitionOutcome, config: RewardConfig) -> float:
    """Return a dimensionless reward for one scheduling transition."""

    delivered_mbit = outcome.delivered_bits / 1e6
    fairness_gain = outcome.fairness_after - outcome.fairness_before
    reward = (
        config.delivered_mbit_weight * delivered_mbit
        - config.deadline_drop_weight * outcome.deadline_drops
        - config.failed_attempt_weight * outcome.failed_attempts
        - config.scheduled_outage_weight * int(outcome.scheduled_outage)
        - config.switch_weight * int(outcome.switched_vehicle)
        + config.fairness_gain_weight * fairness_gain
    )
    return float(reward)
