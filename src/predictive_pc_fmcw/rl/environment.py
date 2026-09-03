from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..scheduling.base import SchedulerContext
from .reward import RewardConfig, TransitionOutcome, compute_reward
from .state import ObservationConfig, build_observation


@dataclass(frozen=True)
class EnvironmentStep:
    """One transition produced by the shared packet-simulation backend."""

    next_context: SchedulerContext | None
    outcome: TransitionOutcome
    terminated: bool
    info: dict[str, object]


class SchedulingTransitionBackend(Protocol):
    """Adapter implemented by the packet simulator, not by an RL algorithm."""

    @property
    def vehicle_count(self) -> int: ...

    def reset(self, seed: int) -> SchedulerContext: ...

    def step(self, vehicle: int | None) -> EnvironmentStep: ...


@dataclass(frozen=True)
class ActionSpec:
    """Discrete action mapping: vehicles 0..N-1 plus one explicit no-op."""

    vehicles: int

    def __post_init__(self) -> None:
        if self.vehicles <= 0:
            raise ValueError("vehicles must be positive")

    @property
    def no_op(self) -> int:
        return self.vehicles

    @property
    def size(self) -> int:
        return self.vehicles + 1

    def decode(self, action: int) -> int | None:
        if not 0 <= int(action) < self.size:
            raise ValueError(f"action {action} is outside [0, {self.size})")
        return None if int(action) == self.no_op else int(action)


class RLSchedulingEnv:
    """Gym-style scheduling environment without a hard Gymnasium dependency.

    Observation/action/reward semantics live here. Physical link evolution,
    packet queues and delivery realization remain in the shared simulation
    backend so RL and heuristic policies can be compared on paired traces.
    """

    def __init__(
        self,
        backend: SchedulingTransitionBackend,
        observation_config: ObservationConfig,
        reward_config: RewardConfig | None = None,
        *,
        allow_no_op_with_backlog: bool = True,
    ) -> None:
        self.backend = backend
        self.observation_config = observation_config
        self.reward_config = reward_config or RewardConfig()
        self.action_spec = ActionSpec(backend.vehicle_count)
        self.allow_no_op_with_backlog = allow_no_op_with_backlog
        self._context: SchedulerContext | None = None

    @property
    def context(self) -> SchedulerContext:
        if self._context is None:
            raise RuntimeError("environment must be reset before use")
        return self._context

    def action_mask(self) -> NDArray[np.bool_]:
        """Return valid actions for the current causal state."""

        eligible = self.context.queue_lengths > 0
        mask = np.zeros(self.action_spec.size, dtype=bool)
        mask[: self.action_spec.vehicles] = eligible
        mask[self.action_spec.no_op] = (
            self.allow_no_op_with_backlog or not bool(np.any(eligible))
        )
        return mask

    def reset(self, *, seed: int) -> tuple[NDArray[np.float32], dict[str, object]]:
        context = self.backend.reset(seed)
        if context.oracle_forecast:
            raise ValueError(
                "RL training/evaluation cannot consume oracle future state"
            )
        if context.vehicles != self.action_spec.vehicles:
            raise ValueError("backend vehicle count changed across reset")
        self._context = context
        observation = build_observation(context, self.observation_config)
        return observation, {"action_mask": self.action_mask().copy()}

    def step(
        self, action: int
    ) -> tuple[
        NDArray[np.float32],
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        action = int(action)
        mask = self.action_mask()
        if not 0 <= action < mask.size or not bool(mask[action]):
            raise ValueError(f"masked or invalid RL action: {action}")

        vehicle = self.action_spec.decode(action)
        transition = self.backend.step(vehicle)
        reward = compute_reward(transition.outcome, self.reward_config)

        info = dict(transition.info)
        info["selected_vehicle"] = vehicle
        info["reward_terms"] = {
            "delivered_bits": transition.outcome.delivered_bits,
            "deadline_drops": transition.outcome.deadline_drops,
            "failed_attempts": transition.outcome.failed_attempts,
            "scheduled_outage": transition.outcome.scheduled_outage,
            "switched_vehicle": transition.outcome.switched_vehicle,
            "fairness_before": transition.outcome.fairness_before,
            "fairness_after": transition.outcome.fairness_after,
        }

        if transition.terminated:
            terminal = np.zeros_like(
                build_observation(self.context, self.observation_config)
            )
            self._context = None
            info["action_mask"] = np.zeros(self.action_spec.size, dtype=bool)
            return terminal, reward, True, False, info

        if transition.next_context is None:
            raise ValueError("non-terminal transition requires next_context")
        if transition.next_context.oracle_forecast:
            raise ValueError("RL transition exposed oracle future state")
        self._context = transition.next_context
        observation = build_observation(self._context, self.observation_config)
        info["action_mask"] = self.action_mask().copy()
        return observation, reward, False, False, info
