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
    traffic_class: str = "best_effort"


@dataclass(frozen=True)
class TrafficTrace:
    arrivals: NDArray[np.int64]
    deadlines: tuple[tuple[tuple[int, ...], ...], ...]
    classes: tuple[tuple[tuple[str, ...], ...], ...]
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
    class_rows: list[tuple[tuple[str, ...], ...]] = []
    for slot in range(slots):
        vehicle_rows: list[tuple[int, ...]] = []
        vehicle_classes: list[tuple[str, ...]] = []
        for vehicle in range(vehicles):
            count = int(arrivals[slot, vehicle])
            if config.traffic_class_mode == "urgent_bulk":
                urgent = rng.random(count) < config.urgent_fraction
                classes = np.where(urgent, "urgent", "bulk")
                urgent_slots = max(
                    1, int(round(config.urgent_deadline_s / slot_duration_s))
                )
                bulk_slots = max(
                    1, int(round(config.bulk_deadline_s / slot_duration_s))
                )
                deadlines = slot + np.where(urgent, urgent_slots, bulk_slots)
            else:
                jitter = rng.integers(
                    -jitter_slots,
                    jitter_slots + 1,
                    size=count,
                )
                deadlines = slot + np.maximum(1, base_deadline_slots + jitter)
                classes = np.full(count, "best_effort")
            vehicle_rows.append(tuple(int(value) for value in deadlines))
            vehicle_classes.append(tuple(str(value) for value in classes))
        deadline_rows.append(tuple(vehicle_rows))
        class_rows.append(tuple(vehicle_classes))
    success_uniforms = rng.random(
        (slots, vehicles, max(1, nominal_capacity_packets)), dtype=np.float64
    )
    return TrafficTrace(
        arrivals=arrivals.astype(np.int64),
        deadlines=tuple(deadline_rows),
        classes=tuple(class_rows),
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
        self.class_generated = {
            name: np.zeros(vehicles, dtype=np.int64)
            for name in ("urgent", "bulk", "best_effort")
        }
        self.class_overflow_dropped = {
            name: np.zeros(vehicles, dtype=np.int64)
            for name in self.class_generated
        }
        self.class_deadline_dropped = {
            name: np.zeros(vehicles, dtype=np.int64)
            for name in self.class_generated
        }
        self._next_packet_id = 0

    def add_arrivals(
        self,
        slot: int,
        deadlines: tuple[tuple[int, ...], ...],
        classes: tuple[tuple[str, ...], ...] | None = None,
    ) -> None:
        class_rows = classes or tuple(
            tuple("best_effort" for _ in row) for row in deadlines
        )
        for vehicle, (vehicle_deadlines, vehicle_classes) in enumerate(
            zip(deadlines, class_rows, strict=True)
        ):
            if len(vehicle_deadlines) != len(vehicle_classes):
                raise ValueError("Traffic classes must align with packet deadlines.")
            self.generated[vehicle] += len(vehicle_deadlines)
            for traffic_class in vehicle_classes:
                self.class_generated[traffic_class][vehicle] += 1
            available = max(0, self.max_packets - len(self.queues[vehicle]))
            accepted = vehicle_deadlines[:available]
            accepted_classes = vehicle_classes[:available]
            self.overflow_dropped[vehicle] += len(vehicle_deadlines) - len(accepted)
            for traffic_class in vehicle_classes[available:]:
                self.class_overflow_dropped[traffic_class][vehicle] += 1
            for deadline, traffic_class in zip(
                accepted, accepted_classes, strict=True
            ):
                self.queues[vehicle].append(
                    Packet(
                        packet_id=self._next_packet_id,
                        vehicle=vehicle,
                        arrival_slot=slot,
                        deadline_slot=int(deadline),
                        traffic_class=traffic_class,
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
                    self.class_deadline_dropped[packet.traffic_class][vehicle] += 1
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
