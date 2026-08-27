from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import LinkConfig


@dataclass(frozen=True)
class LinkState:
    distance_m: float
    bearing_rad: float
    received_power_w: float
    snr_db: float
    ber: float
    per: float
    goodput_bps: float
    outage: bool


class LinkModel:
    """Calibrated, geometry-dependent PC-FMCW/DPSK optical link abstraction.

    Absolute optical noise parameters are not claimed as measured. The model is
    anchored by a configurable reference SNR and preserves physically expected
    distance, atmospheric and pointing monotonicity.
    """

    def __init__(self, config: LinkConfig):
        self.config = config
        self._reference_gain = float(
            self._relative_gain(config.reference_distance_m, 0.0)
        )
        if self._reference_gain <= 0:
            raise ValueError("Reference geometry must define a usable link.")

    def _relative_gain(
        self, distance_m: ArrayLike, bearing_rad: ArrayLike
    ) -> NDArray[np.float64]:
        distance = np.maximum(np.asarray(distance_m, dtype=np.float64), 0.5)
        bearing = np.asarray(bearing_rad, dtype=np.float64)
        divergence = np.deg2rad(self.config.beam_divergence_half_angle_deg)
        sigma = np.deg2rad(self.config.pointing_sigma_deg)
        fov_half = np.deg2rad(self.config.field_of_view_deg / 2)
        footprint_area = np.pi * np.maximum(distance * np.tan(divergence), 0.05) ** 2
        geometric = 1.0 / footprint_area
        atmospheric = np.exp(
            -self.config.atmospheric_attenuation_per_m * distance
        )
        pointing = np.exp(-0.5 * (bearing / sigma) ** 2)
        inside_fov = np.abs(bearing) <= fov_half
        return geometric * atmospheric * pointing * inside_fov

    def received_power_w(
        self, distance_m: ArrayLike, bearing_rad: ArrayLike
    ) -> NDArray[np.float64]:
        gain = self._relative_gain(distance_m, bearing_rad)
        return np.maximum(
            gain / self._reference_gain * 1e-6,
            self.config.min_received_power_w,
        )

    def snr_linear(
        self, distance_m: ArrayLike, bearing_rad: ArrayLike
    ) -> NDArray[np.float64]:
        gain = self._relative_gain(distance_m, bearing_rad)
        reference_snr = 10 ** (self.config.reference_snr_db / 10)
        return np.maximum(reference_snr * gain / self._reference_gain, 0.0)

    @staticmethod
    def dbpsk_ber(snr_linear: ArrayLike) -> NDArray[np.float64]:
        gamma = np.maximum(np.asarray(snr_linear, dtype=np.float64), 0.0)
        return np.clip(0.5 * np.exp(-gamma), 1e-15, 0.5)

    def packet_error_rate(self, ber: ArrayLike) -> NDArray[np.float64]:
        bit_error = np.clip(np.asarray(ber, dtype=np.float64), 0.0, 1.0)
        return -np.expm1(self.config.packet_bits * np.log1p(-bit_error))

    def evaluate_arrays(
        self, distance_m: ArrayLike, bearing_rad: ArrayLike
    ) -> dict[str, NDArray[np.float64]]:
        distance = np.asarray(distance_m, dtype=np.float64)
        bearing = np.asarray(bearing_rad, dtype=np.float64)
        snr_linear = self.snr_linear(distance, bearing)
        snr_db = 10 * np.log10(np.maximum(snr_linear, 1e-15))
        ber = self.dbpsk_ber(snr_linear)
        per = self.packet_error_rate(ber)
        goodput = self.config.data_rate_bps * (1 - per)
        outage = ber > self.config.outage_ber_threshold
        return {
            "received_power_w": self.received_power_w(distance, bearing),
            "snr_linear": snr_linear,
            "snr_db": snr_db,
            "ber": ber,
            "per": per,
            "goodput_bps": goodput,
            "outage": outage,
        }

    def evaluate(self, distance_m: float, bearing_rad: float) -> LinkState:
        values = self.evaluate_arrays(distance_m, bearing_rad)
        return LinkState(
            distance_m=float(distance_m),
            bearing_rad=float(bearing_rad),
            received_power_w=float(values["received_power_w"]),
            snr_db=float(values["snr_db"]),
            ber=float(values["ber"]),
            per=float(values["per"]),
            goodput_bps=float(values["goodput_bps"]),
            outage=bool(values["outage"]),
        )

    def capacity_packets(self, slot_duration_s: float) -> int:
        bits = (
            self.config.data_rate_bps
            * slot_duration_s
            * self.config.resource_fraction
        )
        return max(1, int(bits // self.config.packet_bits))

    def link_lifetime_steps(
        self, distances_m: ArrayLike, bearings_rad: ArrayLike
    ) -> NDArray[np.int64]:
        values = self.evaluate_arrays(distances_m, bearings_rad)
        outage = np.asarray(values["outage"], dtype=bool)
        if outage.ndim != 2:
            raise ValueError("Expected link traces with shape (vehicles, horizon).")
        horizon = outage.shape[1]
        first = np.argmax(outage, axis=1)
        has_outage = np.any(outage, axis=1)
        return np.where(has_outage, first, horizon).astype(np.int64)

