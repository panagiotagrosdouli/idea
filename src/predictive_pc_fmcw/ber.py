from __future__ import annotations

import csv
from dataclasses import dataclass, replace
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
    ber_upper_95: float
    ber_for_lut: float
    receiver: str = "symbol_level_dbpsk"
    snr_semantics: str = "ebn0_db"


def _wilson_upper(errors: int, bits: int, z: float = 1.6448536269514722) -> float:
    proportion = errors / bits
    denominator = 1.0 + z**2 / bits
    center = proportion + z**2 / (2 * bits)
    radius = z * np.sqrt(
        proportion * (1 - proportion) / bits + z**2 / (4 * bits**2)
    )
    return float(min(1.0, (center + radius) / denominator))


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
        simulated = errors / bits
        upper = _wilson_upper(errors, bits)
        points.append(
            BERPoint(
                ebn0_db=float(snr_db),
                simulated_ber=simulated,
                theoretical_ber=float(0.5 * np.exp(-gamma)),
                bits=bits,
                errors=errors,
                ber_upper_95=upper,
                ber_for_lut=simulated if errors > 0 else upper,
            )
        )
    return _monotone_lut_values(points)


def _monotone_lut_values(points: list[BERPoint]) -> list[BERPoint]:
    """Conservatively enforce the physical BER monotonicity used by the LUT.

    Raw Monte Carlo estimates remain untouched in ``simulated_ber``.  A reverse
    cumulative maximum raises lower-SNR LUT values when needed, rather than
    hiding an observed receiver failure at a higher SNR.
    """

    if not points:
        return []
    snr = np.asarray([point.ebn0_db for point in points], dtype=np.float64)
    values = np.asarray([point.ber_for_lut for point in points], dtype=np.float64)
    order = np.argsort(snr)
    monotone_sorted = np.maximum.accumulate(values[order][::-1])[::-1]
    monotone = np.empty_like(monotone_sorted)
    monotone[order] = monotone_sorted
    return [
        replace(point, ber_for_lut=float(value))
        for point, value in zip(points, monotone, strict=True)
    ]


def simulate_dbpsk_ber_adaptive(
    ebn0_db: ArrayLike,
    min_bits: int = 100_000,
    max_bits: int = 1_000_000,
    target_errors: int = 200,
    chunk_bits: int = 100_000,
    seed: int = 20260827,
) -> list[BERPoint]:
    """Estimate DBPSK BER with bounded, error-targeted Monte Carlo effort.

    Zero-error points retain an explicit one-sided Wilson upper bound. The LUT
    uses that bound instead of an arbitrary numerical floor, so high-SNR points
    are represented as measurement-limited rather than falsely exact.
    """

    if not 1_000 <= min_bits <= max_bits:
        raise ValueError("Require 1000 <= min_bits <= max_bits.")
    if target_errors < 1 or chunk_bits < 1_000:
        raise ValueError("target_errors and chunk_bits must be positive.")
    root = np.random.SeedSequence(seed)
    grid = np.asarray(ebn0_db, dtype=np.float64).reshape(-1)
    children = root.spawn(grid.size)
    points: list[BERPoint] = []
    for snr_db, child in zip(grid, children, strict=True):
        rng = np.random.default_rng(child)
        gamma = 10 ** (float(snr_db) / 10)
        sigma = np.sqrt(1 / (2 * gamma))
        total_bits = 0
        total_errors = 0
        while total_bits < max_bits and (
            total_bits < min_bits or total_errors < target_errors
        ):
            count = min(chunk_bits, max_bits - total_bits)
            payload = rng.integers(0, 2, size=count, dtype=np.int8)
            differential = np.empty(count + 1, dtype=np.complex128)
            differential[0] = 1.0
            differential[1:] = np.cumprod(1 - 2 * payload).astype(np.complex128)
            noise = sigma * (
                rng.standard_normal(count + 1)
                + 1j * rng.standard_normal(count + 1)
            )
            received = differential + noise
            detected = (
                np.real(received[1:] * np.conj(received[:-1])) < 0
            ).astype(np.int8)
            total_errors += int(np.count_nonzero(detected != payload))
            total_bits += count
        simulated = total_errors / total_bits
        upper = _wilson_upper(total_errors, total_bits)
        points.append(
            BERPoint(
                ebn0_db=float(snr_db),
                simulated_ber=simulated,
                theoretical_ber=float(0.5 * np.exp(-gamma)),
                bits=total_bits,
                errors=total_errors,
                ber_upper_95=upper,
                ber_for_lut=simulated if total_errors > 0 else upper,
            )
        )
    return _monotone_lut_values(points)


def simulate_part_a_notebook_receiver_ber(
    ebn0_db: ArrayLike,
    bits: int = 100_000,
    seed: int = 20260827,
    *,
    chirp_bandwidth_hz: float = 10e9,
    chirp_duration_s: float = 10e-6,
    data_rate_bps: float = 1e9,
    fast_time_samples: int = 131_072,
    fft_zeropad_factor: int = 8,
) -> list[BERPoint]:
    """Run the supplied Part-A FFT-carrier/DPSK receiver over an SNR grid.

    The waveform timing, chirp phase, per-symbol FFT carrier extraction,
    parabolic peak refinement, carrier-rotation compensation and differential
    decision follow the supplied ``ISCAI_PC_FMCW.ipynb`` communication branch.
    The SNR argument follows the notebook's waveform-sample SNR definition.
    ``BERPoint.ebn0_db`` is retained as the legacy LUT-axis field name, while
    the emitted ``snr_semantics`` column prevents it from being mistaken for
    symbol ``Eb/N0``.
    """

    if bits < 1_000:
        raise ValueError("At least 1000 bits are required for a meaningful LUT.")
    if (
        chirp_bandwidth_hz <= 0
        or chirp_duration_s <= 0
        or data_rate_bps <= 0
        or fast_time_samples < 2
        or fft_zeropad_factor < 1
    ):
        raise ValueError("Part-A waveform and FFT parameters must be positive.")

    sample_rate_hz = fast_time_samples / chirp_duration_s
    symbol_duration_s = 1.0 / data_rate_bps
    time_s = np.arange(fast_time_samples, dtype=np.float64) / sample_rate_hz
    symbols_per_chirp = int(np.floor(chirp_duration_s / symbol_duration_s))
    symbol_edges = np.searchsorted(
        time_s,
        np.arange(symbols_per_chirp + 1, dtype=np.float64) * symbol_duration_s,
        side="left",
    )
    starts = symbol_edges[:-1]
    stops = symbol_edges[1:]
    valid_symbols = stops > starts
    starts = starts[valid_symbols]
    stops = stops[valid_symbols]
    lengths = stops - starts
    symbols_per_chirp = int(starts.size)
    if symbols_per_chirp < 2:
        raise ValueError("Part-A timing produced fewer than two valid symbols.")

    max_length = int(lengths.max())
    local_indices = np.arange(max_length, dtype=np.int64)[None, :]
    sample_indices = starts[:, None] + local_indices
    sample_mask = sample_indices < stops[:, None]
    sample_indices = np.minimum(sample_indices, fast_time_samples - 1)
    centers = 0.5 * (starts + stops - 1)
    centered_indices = sample_indices - centers[:, None]
    fft_length = 1 << int(
        np.ceil(np.log2(fft_zeropad_factor * max_length))
    )

    symbol_index_per_sample = np.searchsorted(
        np.arange(symbols_per_chirp + 1, dtype=np.float64) * symbol_duration_s,
        time_s,
        side="right",
    ) - 1
    symbol_index_per_sample = np.clip(
        symbol_index_per_sample, 0, symbols_per_chirp - 1
    )
    chirp_slope_hz_per_s = chirp_bandwidth_hz / chirp_duration_s
    chirp_phase = np.pi * chirp_slope_hz_per_s * time_s**2
    decisions_per_chirp = symbols_per_chirp - 1
    grid = np.asarray(ebn0_db, dtype=np.float64).reshape(-1)
    children = np.random.SeedSequence(seed).spawn(grid.size)
    points: list[BERPoint] = []

    for snr_db, child in zip(grid, children, strict=True):
        rng = np.random.default_rng(child)
        gamma = 10 ** (float(snr_db) / 10)
        # Match the supplied notebook: signal power is one and waveform noise
        # power is signal_power / SNR before FFT carrier extraction.
        noise_power = 1.0 / gamma
        noise_sigma = np.sqrt(noise_power / 2.0)
        errors = 0
        evaluated = 0
        while evaluated < bits:
            payload = rng.integers(0, 2, size=symbols_per_chirp, dtype=np.int8)
            symbol_phase = np.mod(np.pi * np.cumsum(payload), 2.0 * np.pi)
            transmit = np.exp(
                1j
                * (
                    chirp_phase
                    + symbol_phase[symbol_index_per_sample]
                )
            )
            noise = noise_sigma * (
                rng.standard_normal(fast_time_samples)
                + 1j * rng.standard_normal(fast_time_samples)
            )
            receive = transmit + noise
            segments = receive[sample_indices] * sample_mask
            spectrum = np.fft.fft(segments, n=fft_length, axis=1)
            magnitude = np.abs(spectrum)
            peaks = np.argmax(magnitude, axis=1)
            delta = np.zeros(symbols_per_chirp, dtype=np.float64)
            refinable = (peaks > 0) & (peaks < fft_length - 1)
            rows = np.flatnonzero(refinable)
            if rows.size:
                peak_rows = peaks[rows]
                left = np.log(magnitude[rows, peak_rows - 1] + 1e-12)
                center = np.log(magnitude[rows, peak_rows] + 1e-12)
                right = np.log(magnitude[rows, peak_rows + 1] + 1e-12)
                denominator = left - 2.0 * center + right
                stable = np.abs(denominator) > 1e-12
                refined = np.zeros(rows.size, dtype=np.float64)
                refined[stable] = (
                    0.5 * (left[stable] - right[stable]) / denominator[stable]
                )
                delta[rows] = np.clip(refined, -0.5, 0.5)
            cycles_per_sample = (peaks + delta) / fft_length
            # Use the same continuous alias branch for both carrier projection
            # and midpoint phase compensation.  Projecting with wrapped FFT
            # frequencies but compensating with their unwrapped equivalents is
            # not representation-invariant when adjacent symbol centres are a
            # half-integer number of samples apart (the 13/14-sample timing
            # pattern used here).  A one-cycle/sample branch difference can
            # otherwise introduce a spurious pi rotation and invert a fixed
            # subset of DPSK decisions for an entire chirp.
            unwrapped = (
                np.unwrap(2.0 * np.pi * cycles_per_sample)
                / (2.0 * np.pi)
            )
            projection = np.exp(
                -1j
                * 2.0
                * np.pi
                * unwrapped[:, None]
                * centered_indices
            )
            coefficients = (
                np.sum(segments * projection * sample_mask, axis=1) / lengths
            )
            differential = coefficients[1:] * np.conj(coefficients[:-1])
            carrier_midpoint = 0.5 * (unwrapped[1:] + unwrapped[:-1])
            carrier_step = 2.0 * np.pi * carrier_midpoint * np.diff(centers)
            observations = differential * np.exp(-1j * carrier_step)
            detected = np.real(observations) < 0.0
            reference = payload[1:].astype(bool)
            count = min(decisions_per_chirp, bits - evaluated)
            errors += int(np.count_nonzero(detected[:count] != reference[:count]))
            evaluated += count

        simulated = errors / evaluated
        upper = _wilson_upper(errors, evaluated)
        points.append(
            BERPoint(
                ebn0_db=float(snr_db),
                simulated_ber=simulated,
                theoretical_ber=float(0.5 * np.exp(-gamma)),
                bits=evaluated,
                errors=errors,
                ber_upper_95=upper,
                ber_for_lut=simulated if errors > 0 else upper,
                receiver="supplied_part_a_fft_dpsk",
                snr_semantics="waveform_sample_snr_db",
            )
        )
    return _monotone_lut_values(points)


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
            [
                float(row.get("ber_for_lut") or row["simulated_ber"])
                for row in rows
            ],
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
