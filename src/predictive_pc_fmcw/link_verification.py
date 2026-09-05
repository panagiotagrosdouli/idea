from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_MIN_BITS_PER_POINT = 250_000
CANONICAL_STABILITY_SNR_DB = {5.0, 7.0, 8.0, 10.0}


def _material_raw_increases(
    snr: np.ndarray,
    simulated: np.ndarray,
    *,
    minimum_absolute_increase: float = 0.01,
    minimum_ratio: float = 1.5,
) -> list[dict[str, float | int]]:
    """Find adjacent raw-BER reversals large enough to require investigation.

    The Part-A decisions are clustered within chirps, so bit-level binomial
    intervals are not valid evidence of statistical significance. This gate is
    deliberately an effect-size diagnostic. Statistical inference belongs to
    a chirp-cluster-aware artifact.
    """
    increases: list[dict[str, float | int]] = []
    for index in range(1, len(snr)):
        previous = float(simulated[index - 1])
        current = float(simulated[index])
        absolute_increase = current - previous
        ratio = current / max(previous, np.finfo(np.float64).eps)
        if absolute_increase >= minimum_absolute_increase and ratio >= minimum_ratio:
            increases.append(
                {
                    "lower_snr_db": float(snr[index - 1]),
                    "higher_snr_db": float(snr[index]),
                    "lower_snr_ber": previous,
                    "higher_snr_ber": current,
                    "absolute_increase": absolute_increase,
                    "increase_ratio": ratio,
                }
            )
    return increases


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


def _paired_diagnostic_clears_reversal(
    diagnostic: dict[str, Any],
    material_increases: list[dict[str, float | int]],
) -> bool:
    pairs = {
        (item["lower_snr_db"], item["higher_snr_db"])
        for item in material_increases
    }
    diagnostic_pair = (
        diagnostic.get("lower_snr_db"),
        diagnostic.get("higher_snr_db"),
    )
    return bool(
        pairs == {diagnostic_pair}
        and diagnostic.get("schema") == "paired_chirp_ber_reversal_v1"
        and diagnostic.get("sampling_unit") == "independent_chirp"
        and diagnostic.get("common_random_numbers_within_pair") is True
        and int(diagnostic.get("trials", 0)) >= 100
        and int(diagnostic.get("bootstrap_repetitions", 0)) >= 10_000
        and diagnostic.get("material_reversal_supported") is False
    )


def _stability_diagnostic_valid(diagnostic: dict[str, Any]) -> bool:
    rows = diagnostic.get("rows")
    if not isinstance(rows, list) or not rows:
        return False
    if not (
        diagnostic.get("method")
        == "independent_one_chirp_trials_with_cluster_bootstrap"
        and diagnostic.get("snr_semantics") == "waveform_sample_snr_db"
        and int(diagnostic.get("trials_per_snr", 0)) >= 50
        and 1_000 <= int(diagnostic.get("decisions_per_trial", 0)) <= 9_999
        and int(diagnostic.get("bootstrap_resamples", 0)) >= 10_000
        and 0.0 < float(diagnostic.get("catastrophic_ber_threshold", 0.0)) <= 0.05
    ):
        return False

    try:
        snr_values = {float(row["snr_db"]) for row in rows}
        row_checks = [
            int(row.get("independent_chirp_trials", 0)) >= 50
            and 1_000 <= int(row.get("decisions_per_trial", 0)) <= 9_999
            and int(row.get("catastrophic_chirps", -1)) == 0
            and float(row.get("max_chirp_ber", 1.0))
            < float(diagnostic["catastrophic_ber_threshold"])
            for row in rows
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return CANONICAL_STABILITY_SNR_DB.issubset(snr_values) and all(row_checks)


def _diagnostic_summary(diagnostic: dict[str, Any]) -> dict[str, Any]:
    if (
        diagnostic.get("method")
        == "independent_one_chirp_trials_with_cluster_bootstrap"
    ):
        row_keys = (
            "snr_db",
            "independent_chirp_trials",
            "decisions_per_trial",
            "mean_ber",
            "median_ber",
            "std_ber_across_chirps",
            "p95_chirp_ber",
            "max_chirp_ber",
            "cluster_bootstrap_ci_95",
            "catastrophic_chirps",
            "catastrophic_chirp_rate",
        )
        return {
            "method": diagnostic.get("method"),
            "snr_semantics": diagnostic.get("snr_semantics"),
            "seed": diagnostic.get("seed"),
            "trials_per_snr": diagnostic.get("trials_per_snr"),
            "decisions_per_trial": diagnostic.get("decisions_per_trial"),
            "bootstrap_resamples": diagnostic.get("bootstrap_resamples"),
            "catastrophic_ber_threshold": diagnostic.get(
                "catastrophic_ber_threshold"
            ),
            "rows": [
                {key: row.get(key) for key in row_keys}
                for row in diagnostic.get("rows", [])
            ],
        }

    excluded = {"trial_lower_ber", "trial_higher_ber"}
    return {key: value for key, value in diagnostic.items() if key not in excluded}


def verify_lut(
    path: str | Path,
    *,
    minimum_bits_per_point: int = CANONICAL_MIN_BITS_PER_POINT,
    chirp_diagnostic_path: str | Path | None = None,
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
    simulated = np.asarray(
        [float(row["simulated_ber"]) for row in rows], dtype=np.float64
    )
    receivers = {row["receiver"] for row in rows}
    semantics = {row["snr_semantics"] for row in rows}
    material_increases = _material_raw_increases(snr, simulated)

    diagnostic: dict[str, Any] | None = None
    diagnostic_valid = True
    diagnostic_kind = None
    diagnostic_clears_reversals = not material_increases
    if chirp_diagnostic_path is not None:
        diagnostic_source = Path(chirp_diagnostic_path)
        diagnostic = json.loads(diagnostic_source.read_text(encoding="utf-8"))
        if diagnostic.get("schema") == "paired_chirp_ber_reversal_v1":
            diagnostic_kind = "paired_reversal"
            diagnostic_valid = _paired_diagnostic_clears_reversal(
                diagnostic, material_increases
            )
            if material_increases:
                diagnostic_clears_reversals = diagnostic_valid
        elif (
            diagnostic.get("method")
            == "independent_one_chirp_trials_with_cluster_bootstrap"
        ):
            diagnostic_kind = "independent_chirp_stability"
            diagnostic_valid = _stability_diagnostic_valid(diagnostic)
        else:
            diagnostic_valid = False

    checks = {
        "required_columns": True,
        "snr_points_31": len(rows) == 31,
        "snr_grid_minus5_to_25": bool(
            len(rows) == 31
            and np.array_equal(snr, np.arange(-5.0, 26.0, dtype=np.float64))
        ),
        "finite_lut": bool(np.isfinite(lut).all()),
        "finite_raw_ber": bool(np.isfinite(simulated).all()),
        "raw_ber_in_probability_range": bool(
            ((0.0 <= simulated) & (simulated <= 0.5)).all()
        ),
        "raw_ber_no_unresolved_material_reversal": diagnostic_clears_reversals,
        "chirp_diagnostic_valid": diagnostic_valid,
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
            "raw_ber_material_reversals": material_increases,
            "raw_ber_material_reversal_thresholds": {
                "minimum_absolute_increase": 0.01,
                "minimum_ratio": 1.5,
            },
            "chirp_diagnostic": (
                {
                    "path": str(chirp_diagnostic_path),
                    "sha256": _sha256(Path(chirp_diagnostic_path)),
                    "kind": diagnostic_kind,
                    "summary": _diagnostic_summary(diagnostic),
                }
                if diagnostic is not None
                else None
            ),
            "checks": checks,
        }
    )
    return report
