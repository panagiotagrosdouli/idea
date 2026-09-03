from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ExperimentConfig
from ..data.scenario import MotionScenario
from ..geometry import range_and_bearing
from ..link import LinkModel
from ..metrics import jains_fairness
from ..predictors import TrajectoryPredictor
from ..scheduling.base import SchedulerContext
from ..simulation.engine import _current_heading, _link_forecast
from ..traffic import PacketQueues, TrafficTrace
from .environment import EnvironmentStep
from .reward import TransitionOutcome


@dataclass(frozen=True)
class BackendEpisodeSummary:
    generated_packets: int
    delivered_packets: int
    deadline_dropped_packets: int
    overflow_dropped_packets: int
    failed_attempts: int
    scheduled_outages: int
    switch_count: int


class SimulationTransitionBackend:
    """Step-wise adapter over the canonical packet/link simulation mechanics.

    This backend deliberately reuses LinkModel, PacketQueues, TrafficTrace and
    the engine's causal link-forecast helper. It never exposes ground-truth
    future link state in SchedulerContext; ground truth is used only to realize
    the current packet outcome, matching the non-RL simulator semantics.
    """

    def __init__(
        self,
        scenario: MotionScenario,
        traffic: TrafficTrace,
        config: ExperimentConfig,
        *,
        forecast_mode: str = "reactive",
        learned_predictor: TrajectoryPredictor | None = None,
    ) -> None:
        if forecast_mode == "oracle":
            raise ValueError("oracle forecast is forbidden for deployable RL")
        self.scenario = scenario
        self.traffic = traffic
        self.config = config
        self.forecast_mode = forecast_mode
        self.learned_predictor = learned_predictor
        self.model = LinkModel(config.link)
        self._slots = min(scenario.evaluation_slots, traffic.arrivals.shape[0])
        if traffic.arrivals.shape[1] != scenario.vehicle_count:
            raise ValueError("Traffic trace vehicle count does not match scenario")
        self._seed = 0
        self._slot = 0
        self._queues: PacketQueues | None = None
        self._delivered_bits = np.zeros(self.vehicle_count, dtype=np.float64)
        self._delivered_packets = np.zeros(self.vehicle_count, dtype=np.int64)
        self._failed_attempts = 0
        self._scheduled_outages = 0
        self._switch_count = 0
        self._previous: int | None = None

    @property
    def vehicle_count(self) -> int:
        return self.scenario.vehicle_count

    @property
    def queues(self) -> PacketQueues:
        if self._queues is None:
            raise RuntimeError("backend must be reset before use")
        return self._queues

    def reset(self, seed: int) -> SchedulerContext:
        self._seed = int(seed)
        self._slot = 0
        self._queues = PacketQueues(
            self.vehicle_count,
            self.config.traffic.max_queue_packets,
        )
        self._delivered_bits = np.zeros(self.vehicle_count, dtype=np.float64)
        self._delivered_packets = np.zeros(self.vehicle_count, dtype=np.int64)
        self._failed_attempts = 0
        self._scheduled_outages = 0
        self._switch_count = 0
        self._previous = None
        return self._prepare_context(self._slot)

    def _prepare_context(self, relative_slot: int) -> SchedulerContext:
        if not 0 <= relative_slot < self._slots:
            raise IndexError("slot outside episode")
        self.queues.add_arrivals(
            relative_slot,
            self.traffic.deadlines[relative_slot],
            self.traffic.classes[relative_slot],
        )
        self.queues.expire(relative_slot)
        time_index = self.scenario.start_index + relative_slot
        heading = _current_heading(self.scenario.ego_positions_xy[: time_index + 1])
        distance, bearing = range_and_bearing(
            self.scenario.vehicle_positions_xy[time_index],
            self.scenario.ego_positions_xy[time_index],
            heading,
        )
        current = self.model.evaluate_arrays(distance, bearing)
        predicted, lifetime, is_oracle = _link_forecast(
            self.scenario,
            time_index,
            self.config.prediction_horizon_steps,
            self.forecast_mode,
            self.model,
            learned_predictor=self.learned_predictor,
            history_noise_std_m=self.config.history_measurement_noise_std_m,
            forecast_noise_std_m=self.config.forecast_position_noise_std_m,
            noise_seed=self._seed + 10_000 * relative_slot,
            sensing_config=self.config.sensing,
        )
        if is_oracle:
            raise ValueError("RL backend received oracle forecast")
        self._current = current
        return SchedulerContext(
            slot=relative_slot,
            queue_lengths=self.queues.lengths(),
            time_to_deadline=self.queues.oldest_time_to_deadline(relative_slot),
            current_goodput_bps=current["goodput_bps"],
            current_outage=current["outage"].astype(bool),
            predicted_goodput_bps=predicted["goodput_bps"],
            predicted_outage=predicted["outage"].astype(bool),
            predicted_lifetime_steps=lifetime,
            delivered_bits=self._delivered_bits.copy(),
            previous_vehicle=self._previous,
            data_rate_bps=self.config.link.data_rate_bps,
            discount=self.config.discount,
            oracle_forecast=False,
        )

    def step(self, vehicle: int | None) -> EnvironmentStep:
        if self._queues is None or not 0 <= self._slot < self._slots:
            raise RuntimeError("backend must be reset before stepping")
        if vehicle is not None and not 0 <= vehicle < self.vehicle_count:
            raise ValueError("selected vehicle is outside vehicle range")

        deadline_before = int(self.queues.deadline_dropped.sum())
        fairness_before = jains_fairness(self._delivered_packets)
        delivered_now = 0
        failed_now = 0
        scheduled_outage = False
        switched = False

        if vehicle is not None:
            if self.queues.lengths()[vehicle] <= 0:
                raise ValueError("cannot schedule a vehicle with an empty queue")
            scheduled_outage = bool(self._current["outage"][vehicle])
            self._scheduled_outages += int(scheduled_outage)
            switched = self._previous is not None and self._previous != vehicle
            self._switch_count += int(switched)
            self._previous = vehicle
            capacity = self.model.capacity_packets(self.config.slot_duration_s)
            attempted = self.queues.pop_attempts(vehicle, capacity)
            uniforms = self.traffic.success_uniforms[
                self._slot,
                vehicle,
                : len(attempted),
            ]
            success = uniforms >= float(self._current["per"][vehicle])
            failed = [
                packet
                for packet, ok in zip(attempted, success, strict=True)
                if not ok
            ]
            successful = [
                packet
                for packet, ok in zip(attempted, success, strict=True)
                if ok
            ]
            self.queues.requeue_failed(vehicle, failed)
            failed_now = len(failed)
            delivered_now = len(successful)
            self._failed_attempts += failed_now
            self._delivered_packets[vehicle] += delivered_now
            self._delivered_bits[vehicle] += (
                delivered_now * self.config.link.packet_bits
            )

        fairness_after = jains_fairness(self._delivered_packets)
        self._slot += 1
        terminated = self._slot >= self._slots
        next_context = None if terminated else self._prepare_context(self._slot)
        deadline_after = int(self.queues.deadline_dropped.sum())
        outcome = TransitionOutcome(
            delivered_bits=delivered_now * self.config.link.packet_bits,
            deadline_drops=max(0, deadline_after - deadline_before),
            failed_attempts=failed_now,
            scheduled_outage=scheduled_outage,
            switched_vehicle=switched,
            fairness_before=fairness_before,
            fairness_after=fairness_after,
        )
        info: dict[str, object] = {
            "slot": self._slot - 1,
            "queue_packets": self.queues.lengths().copy(),
            "generated_packets": int(self.queues.generated.sum()),
            "delivered_packets": int(self._delivered_packets.sum()),
        }
        if terminated:
            info["episode_summary"] = BackendEpisodeSummary(
                generated_packets=int(self.queues.generated.sum()),
                delivered_packets=int(self._delivered_packets.sum()),
                deadline_dropped_packets=int(self.queues.deadline_dropped.sum()),
                overflow_dropped_packets=int(self.queues.overflow_dropped.sum()),
                failed_attempts=self._failed_attempts,
                scheduled_outages=self._scheduled_outages,
                switch_count=self._switch_count,
            )
        return EnvironmentStep(
            next_context=next_context,
            outcome=outcome,
            terminated=terminated,
            info=info,
        )
