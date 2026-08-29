from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import TrafficConfig


@dataclass(frozen=True)
class Packet:
    packet_id: int
    vehicle: int
    arrival_slot: int
    deadline_slot: int


@dataclass(frozen=True)
class TrafficTrace:
    arrivals: NDArray[np.int64]
    deadlines: tuple[tuple[tuple[int, ...], ...], ...]
    success_uniforms: NDArray[np.float64]


def generate_traffic_trace(
    seed: int,
    slots: int,
    vehicles: int,
    nominal_capacity_packets: int,
    config: TrafficConfig,
    slot_duration_s: float = 0.1,
) -> TrafficTrace:
    if slot_duration_s <= 0:
        raise ValueError("slot_duration_s must be positive.")
    rng = np.random.default_rng(seed)
    mean_total = config.offered_load * nominal_capacity_packets
    weights = rng.dirichlet(np.full(vehicles, 2.0))
    if config.model == "poisson":
        arrivals = rng.poisson(
            mean_total * weights[None, :], size=(slots, vehicles)
        )
    elif config.model == "periodic":
        arrivals = np.zeros((slots, vehicles), dtype=np.int64)
        interval = config.periodic_interval_slots
        for slot in range(0, slots, interval):
            arrivals[slot] = rng.poisson(mean_total * interval * weights)
    elif config.model == "markov_modulated":
        arrivals = _markov_modulated_arrivals(
            rng, slots, mean_total, weights, config
        )
    else:
        per_vehicle = max(
            1,
            int(
                np.ceil(
                    max(1.0, config.offered_load)
                    * nominal_capacity_packets
                    / vehicles
                )
            ),
        )
        arrivals = np.full((slots, vehicles), per_vehicle, dtype=np.int64)
    if config.deadline_s is not None:
        base_deadline_slots = max(
            1, int(round(config.deadline_s / slot_duration_s))
        )
        jitter_seconds = config.deadline_jitter_s or 0.0
        jitter_slots = max(0, int(round(jitter_seconds / slot_duration_s)))
    else:
        base_deadline_slots = config.deadline_slots
        jitter_slots = config.deadline_jitter_slots
    deadline_rows: list[tuple[tuple[int, ...], ...]] = []
    for slot in range(slots):
        vehicle_rows: list[tuple[int, ...]] = []
        for vehicle in range(vehicles):
            count = int(arrivals[slot, vehicle])
            jitter = rng.integers(
                -jitter_slots,
                jitter_slots + 1,
                size=count,
            )
            deadlines = slot + np.maximum(1, base_deadline_slots + jitter)
            vehicle_rows.append(tuple(int(value) for value in deadlines))
        deadline_rows.append(tuple(vehicle_rows))
    success_uniforms = rng.random(
        (slots, vehicles, max(1, nominal_capacity_packets)), dtype=np.float64
    )
    return TrafficTrace(
        arrivals=arrivals.astype(np.int64),
        deadlines=tuple(deadline_rows),
        success_uniforms=success_uniforms,
    )


def _markov_modulated_arrivals(
    rng: np.random.Generator,
    slots: int,
    mean_total: float,
    weights: NDArray[np.float64],
    config: TrafficConfig,
) -> NDArray[np.int64]:
    vehicles = weights.size
    arrivals = np.zeros((slots, vehicles), dtype=np.int64)
    denominator = config.markov_low_to_high + config.markov_high_to_low
    high_probability = (
        config.markov_low_to_high / denominator if denominator > 0 else 0.0
    )
    mean_scale = (
        (1 - high_probability) * config.markov_low_rate_scale
        + high_probability * config.markov_high_rate_scale
    )
    normalization = max(mean_scale, 1e-12)
    high_state = rng.random(vehicles) < high_probability
    for slot in range(slots):
        leave_high = high_state & (
            rng.random(vehicles) < config.markov_high_to_low
        )
        enter_high = (~high_state) & (
            rng.random(vehicles) < config.markov_low_to_high
        )
        high_state = (high_state & ~leave_high) | enter_high
        scale = np.where(
            high_state,
            config.markov_high_rate_scale,
            config.markov_low_rate_scale,
        )
        rate = mean_total * weights * scale / normalization
        arrivals[slot] = rng.poisson(rate)
    return arrivals


class PacketQueues:
    def __init__(self, vehicles: int, max_packets: int):
        self.queues = [deque() for _ in range(vehicles)]
        self.max_packets = max_packets
        self.generated = np.zeros(vehicles, dtype=np.int64)
        self.overflow_dropped = np.zeros(vehicles, dtype=np.int64)
        self.deadline_dropped = np.zeros(vehicles, dtype=np.int64)
        self._next_packet_id = 0

    def add_arrivals(self, slot: int, deadlines: tuple[tuple[int, ...], ...]) -> None:
        for vehicle, vehicle_deadlines in enumerate(deadlines):
            self.generated[vehicle] += len(vehicle_deadlines)
            available = max(0, self.max_packets - len(self.queues[vehicle]))
            accepted = vehicle_deadlines[:available]
            self.overflow_dropped[vehicle] += len(vehicle_deadlines) - len(accepted)
            for deadline in accepted:
                self.queues[vehicle].append(
                    Packet(
                        packet_id=self._next_packet_id,
                        vehicle=vehicle,
                        arrival_slot=slot,
                        deadline_slot=int(deadline),
                    )
                )
                self._next_packet_id += 1

    def expire(self, slot: int) -> None:
        for vehicle, queue in enumerate(self.queues):
            kept: deque[Packet] = deque()
            while queue:
                packet = queue.popleft()
                if packet.deadline_slot < slot:
                    self.deadline_dropped[vehicle] += 1
                else:
                    kept.append(packet)
            self.queues[vehicle] = kept

    def pop_attempts(self, vehicle: int, attempts: int) -> list[Packet]:
        queue = self.queues[vehicle]
        return [queue.popleft() for _ in range(min(attempts, len(queue)))]

    def requeue_failed(self, vehicle: int, packets: list[Packet]) -> None:
        for packet in reversed(packets):
            self.queues[vehicle].appendleft(packet)

    def lengths(self) -> NDArray[np.int64]:
        return np.asarray([len(queue) for queue in self.queues], dtype=np.int64)

    def oldest_time_to_deadline(self, slot: int) -> NDArray[np.float64]:
        return np.asarray(
            [
                max(0, queue[0].deadline_slot - slot) if queue else np.inf
                for queue in self.queues
            ],
            dtype=np.float64,
        )

    def remaining(self) -> NDArray[np.int64]:
        return self.lengths()
