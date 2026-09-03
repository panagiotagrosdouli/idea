from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np

CANONICAL_MIN_BITS_PER_POINT = 250_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_report(source: Path) -> dict[str, object]:
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "required_minimum_bits_per_point": CANONICAL_MIN_BITS_PER_POINT,
        "scientific_disclosure": {
            "absolute_received_power": "normalized/model-based, not measured watts",
            "measured_optical_channel_claimed": False,
            "lut_axis": "waveform-sample SNR in dB; legacy column name ebn0_db",
        },
    }


def verify_lut(
    path: str | Path,
    *,
    minimum_bits_per_point: int = CANONICAL_MIN_BITS_PER_POINT,
) -> dict[str, object]:
    if minimum_bits_per_point < 1_000:
        raise ValueError("minimum_bits_per_point must be at least 1000")

    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "ebn0_db",
        "simulated_ber",
        "bits",
        "errors",
        "ber_upper_95",
        "ber_for_lut",
        "receiver",
        "snr_semantics",
    }
    columns = set(rows[0]) if rows else set()
    required_columns = required.issubset(columns)
    report = _base_report(source)
    report["required_minimum_bits_per_point"] = minimum_bits_per_point
    report["rows"] = len(rows)

    if not required_columns:
        report.update(
            {
                "status": "FAIL",
                "min_bits_per_point": 0,
                "max_bits_per_point": 0,
                "missing_columns": sorted(required - columns),
                "checks": {
                    "required_columns": False,
                    "snr_points_31": len(rows) == 31,
                },
            }
        )
        return report

    snr = np.asarray([float(row["ebn0_db"]) for row in rows], dtype=np.float64)
    lut = np.asarray([float(row["ber_for_lut"]) for row in rows], dtype=np.float64)
    upper = np.asarray([float(row["ber_upper_95"]) for row in rows], dtype=np.float64)
    errors = np.asarray([int(row["errors"]) for row in rows], dtype=np.int64)
    bits = np.asarray([int(row["bits"]) for row in rows], dtype=np.int64)
    receivers = {row["receiver"] for row in rows}
    semantics = {row["snr_semantics"] for row in rows}

    checks = {
        "required_columns": True,
        "snr_points_31": len(rows) == 31,
        "snr_grid_minus5_to_25": bool(
            len(rows) == 31
            and np.array_equal(snr, np.arange(-5.0, 26.0, dtype=np.float64))
        ),
        "finite_lut": bool(np.isfinite(lut).all()),
        "lut_in_probability_range": bool(((0.0 <= lut) & (lut <= 0.5)).all()),
        "monotone_nonincreasing": bool(np.all(np.diff(lut) <= 1e-15)),
        "zero_error_points_use_confidence_bound": bool(
            np.all(lut[errors == 0] >= upper[errors == 0])
        ),
        "minimum_bits_per_point_met": bool(
            len(bits) > 0 and np.all(bits >= minimum_bits_per_point)
        ),
        "part_a_receiver": receivers == {"supplied_part_a_fft_dpsk"},
        "waveform_sample_snr_semantics": semantics == {"waveform_sample_snr_db"},
    }
    passed = all(checks.values())
    report.update(
        {
            "status": "PASS" if passed else "FAIL",
            "min_bits_per_point": int(bits.min()) if len(bits) else 0,
            "max_bits_per_point": int(bits.max()) if len(bits) else 0,
            "missing_columns": [],
            "checks": checks,
        }
    )
    return report
