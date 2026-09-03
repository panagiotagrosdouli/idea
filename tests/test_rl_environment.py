from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from predictive_pc_fmcw.rl.environment import EnvironmentStep, RLSchedulingEnv
from predictive_pc_fmcw.rl.reward import TransitionOutcome
from predictive_pc_fmcw.rl.state import ObservationConfig
from predictive_pc_fmcw.scheduling.base import SchedulerContext


def _context(*, oracle: bool = False, queues: tuple[int, int] = (2, 0)) -> SchedulerContext:
    return SchedulerContext(
        slot=0,
        queue_lengths=np.asarray(queues, dtype=np.int64),
        time_to_deadline=np.asarray([2.0, np.inf], dtype=np.float64),
        current_goodput_bps=np.asarray([8e6, 5e6], dtype=np.float64),
        current_outage=np.asarray([False, False]),
        predicted_goodput_bps=np.asarray([[8e6, 6e6], [5e6, 2e6]], dtype=np.float64),
        predicted_outage=np.asarray([[False, False], [False, True]]),
        predicted_lifetime_steps=np.asarray([2, 1], dtype=np.int64),
        delivered_bits=np.asarray([0.0, 0.0], dtype=np.float64),
        previous_vehicle=None,
        data_rate_bps=10e6,
        discount=0.95,
        oracle_forecast=oracle,
    )


@dataclass
class _Backend:
    terminal: bool = False
    oracle: bool = False
    vehicle_count: int = 2

    def reset(self, seed: int) -> SchedulerContext:
        assert isinstance(seed, int)
        return _context(oracle=self.oracle)

    def step(self, vehicle: int | None) -> EnvironmentStep:
        outcome = TransitionOutcome(
            delivered_bits=1000 if vehicle == 0 else 0,
            deadline_drops=0,
            failed_attempts=0,
            scheduled_outage=False,
            switched_vehicle=False,
            fairness_before=0.0,
            fairness_after=0.5 if vehicle == 0 else 0.0,
        )
        return EnvironmentStep(
            next_context=None if self.terminal else _context(queues=(1, 1)),
            outcome=outcome,
            terminated=self.terminal,
            info={"backend": "fixture"},
        )


def _env(backend: _Backend) -> RLSchedulingEnv:
    return RLSchedulingEnv(
        backend,
        ObservationConfig(
            max_queue_packets=10,
            max_deadline_steps=10,
            prediction_horizon_steps=2,
            include_prediction=True,
        ),
    )


def test_reset_returns_observation_and_mask() -> None:
    env = _env(_Backend())
    observation, info = env.reset(seed=7)
    assert observation.shape == (2, 10)
    assert info["action_mask"].tolist() == [True, False, True]


def test_masked_vehicle_action_is_rejected() -> None:
    env = _env(_Backend())
    env.reset(seed=7)
    try:
        env.step(1)
    except ValueError as exc:
        assert "masked" in str(exc)
    else:
        raise AssertionError("masked action should fail")


def test_step_uses_backend_and_exposes_reward_terms() -> None:
    env = _env(_Backend())
    env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(0)
    assert observation.shape == (2, 10)
    assert reward > 0.0
    assert not terminated
    assert not truncated
    assert info["selected_vehicle"] == 0
    assert info["reward_terms"]["delivered_bits"] == 1000
    assert info["action_mask"].tolist() == [True, True, True]


def test_terminal_transition_keeps_observation_shape() -> None:
    env = _env(_Backend(terminal=True))
    initial, _ = env.reset(seed=7)
    terminal, _, terminated, truncated, _ = env.step(0)
    assert terminated
    assert not truncated
    assert terminal.shape == initial.shape
    assert np.all(terminal == 0.0)


def test_oracle_context_is_rejected() -> None:
    env = _env(_Backend(oracle=True))
    try:
        env.reset(seed=7)
    except ValueError as exc:
        assert "oracle" in str(exc).lower()
    else:
        raise AssertionError("oracle RL context should fail closed")
