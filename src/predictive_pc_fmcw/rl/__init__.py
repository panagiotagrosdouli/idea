"""Reinforcement-learning components for packet scheduling.

The RL package consumes the same causal SchedulerContext as the hand-designed
schedulers and must never expose future ground truth to deployable agents.
"""

from .environment import (
    ActionSpec,
    EnvironmentStep,
    RLSchedulingEnv,
    SchedulingTransitionBackend,
)
from .reward import RewardConfig, TransitionOutcome, compute_reward
from .state import ObservationConfig, build_observation, feature_names

__all__ = [
    "ActionSpec",
    "EnvironmentStep",
    "ObservationConfig",
    "RLSchedulingEnv",
    "RewardConfig",
    "SchedulingTransitionBackend",
    "TransitionOutcome",
    "build_observation",
    "compute_reward",
    "feature_names",
]
