from __future__ import annotations

import numpy as np

from predictive_pc_fmcw.rl import (
    ObservationConfig,
    RewardConfig,
    TransitionOutcome,
    build_observation,
    compute_reward,
    feature_names,
)
from predictive_pc_fmcw.scheduling.base import SchedulerContext


def _context() -> SchedulerContext:
    return SchedulerContext(
        slot=3,
        queue_lengths=np.array([4, 0], dtype=np.int64),
        time_to_deadline=np.array([2.0, np.inf], dtype=np.float64),
        current_goodput_bps=np.array([8e6, 2e6], dtype=np.float64),
        current_outage=np.array([False, True]),
        predicted_goodput_bps=np.array(
            [[7e6, 5e6, 1e6], [3e6, 2e6, 0.0]], dtype=np.float64
        ),
        predicted_outage=np.array(
            [[False, False, True], [False, True, True]], dtype=bool
        ),
        predicted_lifetime_steps=np.array([2, 1], dtype=np.int64),
        delivered_bits=np.array([2e6, 1e6], dtype=np.float64),
        previous_vehicle=0,
        data_rate_bps=10e6,
        discount=0.95,
    )


def test_predictive_observation_is_finite_and_bounded() -> None:
    observation = build_observation(
        _context(),
        ObservationConfig(
            max_queue_packets=8,
            max_deadline_steps=10,
            prediction_horizon_steps=3,
            include_prediction=True,
        ),
    )
    assert observation.shape == (2, len(feature_names(True)))
    assert np.all(np.isfinite(observation))
    assert np.all((observation >= 0.0) & (observation <= 1.0))
    assert observation[0, 0] == 1.0
    assert observation[1, 0] == 0.0


def test_reactive_observation_excludes_predictive_features() -> None:
    observation = build_observation(
        _context(),
        ObservationConfig(
            max_queue_packets=8,
            max_deadline_steps=10,
            prediction_horizon_steps=3,
            include_prediction=False,
        ),
    )
    assert observation.shape == (2, len(feature_names(False)))


def test_reward_prefers_delivery_without_failures() -> None:
    config = RewardConfig()
    clean = TransitionOutcome(
        delivered_bits=2_000_000,
        deadline_drops=0,
        failed_attempts=0,
        scheduled_outage=False,
        switched_vehicle=False,
        fairness_before=0.5,
        fairness_after=0.6,
    )
    bad = TransitionOutcome(
        delivered_bits=0,
        deadline_drops=1,
        failed_attempts=1,
        scheduled_outage=True,
        switched_vehicle=True,
        fairness_before=0.5,
        fairness_after=0.4,
    )
    assert compute_reward(clean, config) > compute_reward(bad, config)
