from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..config import ExperimentConfig
from ..data.scenario import MotionScenario
from ..geometry import heading_from_positions, range_and_bearing
from ..link import LinkModel
from ..metrics import SimulationMetrics, jains_fairness
from ..predictors import (
    ConstantAccelerationPredictor,
    ConstantVelocityPredictor,
    forecast_scenario,
)
from ..predictors import TrajectoryPredictor
from ..scheduling.base import SchedulerContext
from ..scheduling.policies import build_scheduler
from ..traffic import PacketQueues, TrafficTrace


@dataclass(frozen=True)
class SimulationOutput:
    metrics: SimulationMetrics
    selected_vehicle: NDArray[np.int64]
    queue_packets: NDArray[np.int64]
    actual_snr_db: NDArray[np.float64]
    actual_outage: NDArray[np.bool_]
    delivered_packets_by_vehicle: NDArray[np.int64]
    forecast_mode: str


def _current_heading(ego_history: NDArray[np.float64]) -> float:
    delta = ego_history[-1] - ego_history[-2]
    if np.linalg.norm(delta) <= 1e-9:
        return 0.0
    return float(np.arctan2(delta[1], delta[0]))


def _link_forecast(
    scenario: MotionScenario,
    time_index: int,
    horizon: int,
    mode: str,
    model: LinkModel,
    learned_predictor: TrajectoryPredictor | None = None,
) -> tuple[dict[str, NDArray], NDArray[np.int64], bool]:
    predictors = {
        "constant_velocity": ConstantVelocityPredictor(),
        "constant_acceleration": ConstantAccelerationPredictor(),
        "reactive": None,
        "oracle": None,
        "learned": learned_predictor,
    }
    if mode not in predictors:
        raise ValueError(f"Unknown forecast mode: {mode}")
    if mode == "learned":
        if learned_predictor is None:
            raise ValueError("learned forecast mode requires a checkpoint predictor.")
        relative_history = (
            scenario.vehicle_positions_xy[: time_index + 1]
            - scenario.ego_positions_xy[: time_index + 1, None, :]
        ).transpose(1, 0, 2)
        relative_prediction = learned_predictor.predict(
            relative_history, horizon, scenario.dt_s
        )
        ego_history = scenario.ego_positions_xy[: time_index + 1][None, :, :]
        ego_prediction = ConstantVelocityPredictor().predict(
            ego_history, horizon, scenario.dt_s
        )[0]
        vehicle_prediction = relative_prediction + ego_prediction[None, :, :]
        oracle_forecast = False
    else:
        bundle = forecast_scenario(
            scenario.combined_positions(),
            time_index,
            horizon,
            scenario.dt_s,
            predictors[mode],
            oracle=mode == "oracle",
        )
        ego_prediction = bundle.ego_xy
        vehicle_prediction = bundle.vehicle_xy
        oracle_forecast = bundle.oracle
    current_ego = scenario.ego_positions_xy[time_index]
    heading_path = np.concatenate([current_ego[None, :], ego_prediction], axis=0)
    headings = heading_from_positions(heading_path)[1:]
    distances, bearings = range_and_bearing(
        vehicle_prediction,
        ego_prediction[None, :, :],
        headings[None, :],
    )
    values = model.evaluate_arrays(distances, bearings)
    lifetime = model.link_lifetime_steps(distances, bearings)
    return values, lifetime, oracle_forecast


def run_simulation(
    scenario: MotionScenario,
    scheduler_name: str,
    traffic: TrafficTrace,
    config: ExperimentConfig,
    seed: int,
    learned_predictor: TrajectoryPredictor | None = None,
) -> SimulationOutput:
    model = LinkModel(config.link)
    scheduler = build_scheduler(scheduler_name, config.scheduler, seed)
    slots = min(scenario.evaluation_slots, traffic.arrivals.shape[0])
    vehicles = scenario.vehicle_count
    if traffic.arrivals.shape[1] != vehicles:
        raise ValueError("Traffic trace vehicle count does not match scenario.")
    queues = PacketQueues(vehicles, config.traffic.max_queue_packets)
    selected = np.full(slots, -1, dtype=np.int64)
    queue_series = np.zeros((slots, vehicles), dtype=np.int64)
    actual_snr = np.empty((slots, vehicles), dtype=np.float64)
    actual_outage = np.empty((slots, vehicles), dtype=bool)
    delivered_by_vehicle = np.zeros(vehicles, dtype=np.int64)
    delivered_bits = np.zeros(vehicles, dtype=np.float64)
    failed_attempts = 0
    scheduled_outages = 0
    scheduled_slots = 0
    switch_count = 0
    previous: int | None = None
    latencies: list[float] = []
    capacity = model.capacity_packets(config.slot_duration_s)

    for relative_slot in range(slots):
        time_index = scenario.start_index + relative_slot
        queues.add_arrivals(relative_slot, traffic.deadlines[relative_slot])
        queues.expire(relative_slot)
        current_heading = _current_heading(scenario.ego_positions_xy[: time_index + 1])
        distance, bearing = range_and_bearing(
            scenario.vehicle_positions_xy[time_index],
            scenario.ego_positions_xy[time_index],
            current_heading,
        )
        current = model.evaluate_arrays(distance, bearing)
        actual_snr[relative_slot] = current["snr_db"]
        actual_outage[relative_slot] = current["outage"]
        predicted, lifetime, is_oracle = _link_forecast(
            scenario,
            time_index,
            config.prediction_horizon_steps,
            scheduler.forecast_mode,
            model,
            learned_predictor=learned_predictor,
        )
        context = SchedulerContext(
            slot=relative_slot,
            queue_lengths=queues.lengths(),
            time_to_deadline=queues.oldest_time_to_deadline(relative_slot),
            current_goodput_bps=current["goodput_bps"],
            current_outage=current["outage"].astype(bool),
            predicted_goodput_bps=predicted["goodput_bps"],
            predicted_outage=predicted["outage"].astype(bool),
            predicted_lifetime_steps=lifetime,
            delivered_bits=delivered_bits.copy(),
            previous_vehicle=previous,
            data_rate_bps=config.link.data_rate_bps,
            discount=config.discount,
            oracle_forecast=is_oracle,
        )
        decision = scheduler.select(context)
        queue_series[relative_slot] = context.queue_lengths
        if decision.vehicle is None:
            continue
        vehicle = decision.vehicle
        selected[relative_slot] = vehicle
        scheduled_slots += 1
        scheduled_outages += int(current["outage"][vehicle])
        if previous is not None and previous != vehicle:
            switch_count += 1
        previous = vehicle
        attempted = queues.pop_attempts(vehicle, capacity)
        uniforms = traffic.success_uniforms[
            relative_slot, vehicle, : len(attempted)
        ]
        success = uniforms >= float(current["per"][vehicle])
        failed = [packet for packet, ok in zip(attempted, success, strict=True) if not ok]
        queues.requeue_failed(vehicle, failed)
        successful = [
            packet for packet, ok in zip(attempted, success, strict=True) if ok
        ]
        failed_attempts += len(failed)
        delivered_by_vehicle[vehicle] += len(successful)
        delivered_bits[vehicle] += len(successful) * config.link.packet_bits
        latencies.extend(
            (relative_slot - packet.arrival_slot + 1) * config.slot_duration_s
            for packet in successful
        )

    generated = int(queues.generated.sum())
    delivered = int(delivered_by_vehicle.sum())
    deadline_dropped = int(queues.deadline_dropped.sum())
    overflow_dropped = int(queues.overflow_dropped.sum())
    remaining = int(queues.remaining().sum())
    duration = slots * config.slot_duration_s
    latency_ms = np.asarray(latencies, dtype=np.float64) * 1e3
    metrics = SimulationMetrics(
        scheduler=scheduler_name,
        scenario_id=scenario.scenario_id,
        source=scenario.source,
        seed=seed,
        vehicles=vehicles,
        duration_s=duration,
        generated_packets=generated,
        delivered_packets=delivered,
        failed_attempts=failed_attempts,
        deadline_dropped_packets=deadline_dropped,
        overflow_dropped_packets=overflow_dropped,
        remaining_packets=remaining,
        goodput_mbps=float(delivered * config.link.packet_bits / duration / 1e6),
        packet_delivery_ratio=delivered / max(1, generated),
        scheduled_outage_fraction=scheduled_outages / max(1, scheduled_slots),
        availability_outage_fraction=float(actual_outage.mean()),
        mean_latency_ms=float(latency_ms.mean()) if latency_ms.size else float("nan"),
        p95_latency_ms=float(np.quantile(latency_ms, 0.95))
        if latency_ms.size
        else float("nan"),
        deadline_miss_ratio=(deadline_dropped + overflow_dropped) / max(1, generated),
        jain_fairness=jains_fairness(delivered_by_vehicle),
        switch_count=switch_count,
    )
    return SimulationOutput(
        metrics=metrics,
        selected_vehicle=selected,
        queue_packets=queue_series,
        actual_snr_db=actual_snr,
        actual_outage=actual_outage,
        delivered_packets_by_vehicle=delivered_by_vehicle,
        forecast_mode=scheduler.forecast_mode,
    )
