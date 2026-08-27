from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SchedulerContext:
    slot: int
    queue_lengths: NDArray[np.int64]
    time_to_deadline: NDArray[np.float64]
    current_goodput_bps: NDArray[np.float64]
    current_outage: NDArray[np.bool_]
    predicted_goodput_bps: NDArray[np.float64]
    predicted_outage: NDArray[np.bool_]
    predicted_lifetime_steps: NDArray[np.int64]
    delivered_bits: NDArray[np.float64]
    previous_vehicle: int | None
    data_rate_bps: float
    discount: float
    oracle_forecast: bool = False

    @property
    def vehicles(self) -> int:
        return int(self.queue_lengths.size)


@dataclass(frozen=True)
class SchedulerDecision:
    vehicle: int | None
    scores: NDArray[np.float64]
    policy: str


class Scheduler(Protocol):
    name: str
    forecast_mode: str

    def select(self, context: SchedulerContext) -> SchedulerDecision: ...


def eligible_mask(context: SchedulerContext) -> NDArray[np.bool_]:
    return context.queue_lengths > 0


def choose_best(
    scores: NDArray[np.float64],
    eligible: NDArray[np.bool_],
    policy: str,
) -> SchedulerDecision:
    safe = np.where(eligible, scores, -np.inf)
    if not np.any(eligible):
        return SchedulerDecision(None, safe, policy)
    return SchedulerDecision(int(np.argmax(safe)), safe, policy)

