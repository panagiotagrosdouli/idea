from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class BERPoint:
    ebn0_db: float
    simulated_ber: float
    theoretical_ber: float
    bits: int
    errors: int


def simulate_dbpsk_ber(
    ebn0_db: ArrayLike,
    bits: int = 250_000,
    seed: int = 20260827,
) -> list[BERPoint]:
    if bits < 1_000:
        raise ValueError("At least 1000 bits are required for a meaningful LUT.")
    rng = np.random.default_rng(seed)
    snr_grid = np.asarray(ebn0_db, dtype=np.float64).reshape(-1)
    payload = rng.integers(0, 2, size=bits, dtype=np.int8)
    differential = np.empty(bits + 1, dtype=np.complex128)
    differential[0] = 1.0
    differential[1:] = np.cumprod(1 - 2 * payload).astype(np.complex128)
    points: list[BERPoint] = []
    for snr_db in snr_grid:
        gamma = 10 ** (float(snr_db) / 10)
        sigma = np.sqrt(1 / (2 * gamma))
        noise = sigma * (
            rng.standard_normal(bits + 1) + 1j * rng.standard_normal(bits + 1)
        )
        received = differential + noise
        detected = (
            np.real(received[1:] * np.conj(received[:-1])) < 0
        ).astype(np.int8)
        errors = int(np.count_nonzero(detected != payload))
        points.append(
            BERPoint(
                ebn0_db=float(snr_db),
                simulated_ber=errors / bits,
                theoretical_ber=float(0.5 * np.exp(-gamma)),
                bits=bits,
                errors=errors,
            )
        )
    return points


def write_ber_lut(points: list[BERPoint], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BERPoint.__dataclass_fields__)
        writer.writeheader()
        for point in points:
            writer.writerow(point.__dict__)
    return destination


class DPSKLookupTable:
    def __init__(self, ebn0_db: ArrayLike, ber: ArrayLike):
        self.ebn0_db = np.asarray(ebn0_db, dtype=np.float64)
        self.ber = np.asarray(ber, dtype=np.float64)
        if self.ebn0_db.ndim != 1 or self.ber.shape != self.ebn0_db.shape:
            raise ValueError("LUT arrays must be one-dimensional and aligned.")
        order = np.argsort(self.ebn0_db)
        self.ebn0_db = self.ebn0_db[order]
        self.ber = np.clip(self.ber[order], 1e-15, 0.5)

    @classmethod
    def from_csv(cls, path: str | Path) -> DPSKLookupTable:
        with Path(path).open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return cls(
            [float(row["ebn0_db"]) for row in rows],
            [float(row["simulated_ber"]) for row in rows],
        )

    def __call__(self, ebn0_db: ArrayLike) -> NDArray[np.float64]:
        query = np.asarray(ebn0_db, dtype=np.float64)
        log_ber = np.interp(
            query,
            self.ebn0_db,
            np.log10(self.ber),
            left=np.log10(self.ber[0]),
            right=np.log10(self.ber[-1]),
        )
        return 10**log_ber
