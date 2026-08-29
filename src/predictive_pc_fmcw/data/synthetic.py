from __future__ import annotations

import numpy as np

from .scenario import MotionScenario


def generate_synthetic_scenario(
    seed: int,
    slots: int = 120,
    vehicles: int = 5,
    dt_s: float = 0.1,
    history_steps: int = 10,
) -> MotionScenario:
    """Generate reproducible motion with closing links and lane changes.

    The trajectory shapes are synthetic and are used for software validation and
    controlled scheduling ablations. They are not presented as WOMD results.
    """

    if vehicles < 1 or slots < 2 or history_steps < 2:
        raise ValueError("vehicles, slots and history_steps must be positive.")
    rng = np.random.default_rng(seed)
    total = history_steps + slots
    time = np.arange(total, dtype=np.float64) * dt_s
    ego_speed = rng.uniform(11.0, 16.0)
    ego = np.column_stack((ego_speed * time, 0.6 * np.sin(0.08 * time)))
    targets = np.empty((total, vehicles, 2), dtype=np.float64)
    for vehicle in range(vehicles):
        initial_x = rng.uniform(24.0, 105.0) + 9.0 * vehicle
        lane_y = rng.choice([-7.2, -3.6, 0.0, 3.6, 7.2])
        relative_speed = rng.uniform(-7.0, 5.0)
        acceleration = rng.uniform(-0.45, 0.35)
        x_relative = (
            initial_x + relative_speed * time + 0.5 * acceleration * time**2
        )
        lane_change_start = rng.uniform(2.0, max(2.1, time[-1] * 0.72))
        lane_change_sign = rng.choice([-1.0, 1.0])
        lane_change = lane_change_sign * rng.uniform(0.0, 7.5) / (
            1 + np.exp(-1.8 * (time - lane_change_start))
        )
        y_relative = lane_y + lane_change + 0.35 * np.sin(
            0.45 * time + rng.uniform(0, 2 * np.pi)
        )
        targets[:, vehicle, 0] = ego[:, 0] + x_relative
        targets[:, vehicle, 1] = ego[:, 1] + y_relative
    return MotionScenario(
        scenario_id=(
            f"synthetic-{seed}-v{vehicles}-slots{slots}-dt{dt_s:.6f}"
        ),
        timestamps_s=time,
        ego_positions_xy=ego,
        vehicle_positions_xy=targets,
        actor_ids=tuple(f"vehicle-{index}" for index in range(vehicles)),
        start_index=history_steps,
        source="controlled_synthetic_motion",
    )
