from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import ExperimentConfig
from .link import LinkModel
from .predictors import ConstantVelocityPredictor, forecast_scenario


@dataclass(frozen=True)
class ValidationReport:
    status: str
    distance_monotonic: bool
    pointing_monotonic: bool
    ber_monotonic: bool
    causal_forecast_invariant: bool
    oracle_forecast_sensitive: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_validation(config: ExperimentConfig) -> ValidationReport:
    model = LinkModel(config.link)
    distances = np.asarray([20.0, 40.0, 80.0])
    distance_snr = model.evaluate_arrays(distances, np.zeros(3))["snr_db"]
    bearings = np.deg2rad(np.asarray([0.0, 10.0, 20.0]))
    pointing_snr = model.evaluate_arrays(np.full(3, 40.0), bearings)["snr_db"]
    ber = model.dbpsk_ber(np.asarray([0.5, 2.0, 10.0]))

    time = np.arange(8, dtype=float)
    positions = np.zeros((8, 2, 2), dtype=float)
    positions[:, 0, 0] = time
    positions[:, 1, 0] = time + 20
    mutated = positions.copy()
    mutated[5:, 1, 1] = 1_000
    causal_a = forecast_scenario(
        positions, 4, 3, 0.1, ConstantVelocityPredictor(), oracle=False
    )
    causal_b = forecast_scenario(
        mutated, 4, 3, 0.1, ConstantVelocityPredictor(), oracle=False
    )
    oracle_a = forecast_scenario(positions, 4, 3, 0.1, None, oracle=True)
    oracle_b = forecast_scenario(mutated, 4, 3, 0.1, None, oracle=True)
    checks = {
        "distance_monotonic": bool(np.all(np.diff(distance_snr) < 0)),
        "pointing_monotonic": bool(np.all(np.diff(pointing_snr) < 0)),
        "ber_monotonic": bool(np.all(np.diff(ber) < 0)),
        "causal_forecast_invariant": bool(
            np.array_equal(causal_a.vehicle_xy, causal_b.vehicle_xy)
        ),
        "oracle_forecast_sensitive": bool(
            not np.array_equal(oracle_a.vehicle_xy, oracle_b.vehicle_xy)
        ),
    }
    return ValidationReport(
        status="PASS" if all(checks.values()) else "FAIL",
        **checks,
    )

