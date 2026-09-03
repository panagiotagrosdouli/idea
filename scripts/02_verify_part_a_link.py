from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lut(path: str | Path) -> dict[str, object]:
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
    snr = np.asarray([float(row["ebn0_db"]) for row in rows], dtype=np.float64)
    lut = np.asarray([float(row["ber_for_lut"]) for row in rows], dtype=np.float64)
    upper = np.asarray([float(row["ber_upper_95"]) for row in rows], dtype=np.float64)
    errors = np.asarray([int(row["errors"]) for row in rows], dtype=np.int64)
    bits = np.asarray([int(row["bits"]) for row in rows], dtype=np.int64)
    receivers = {row["receiver"] for row in rows}
    semantics = {row["snr_semantics"] for row in rows}

    checks = {
        "required_columns": required.issubset(columns),
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
        "positive_bit_counts": bool(np.all(bits > 0)),
        "part_a_receiver": receivers == {"supplied_part_a_fft_dpsk"},
        "waveform_sample_snr_semantics": semantics == {"waveform_sample_snr_db"},
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "path": str(source),
        "sha256": _sha256(source),
        "rows": len(rows),
        "min_bits_per_point": int(bits.min()) if len(bits) else 0,
        "max_bits_per_point": int(bits.max()) if len(bits) else 0,
        "checks": checks,
        "scientific_disclosure": {
            "absolute_received_power": "normalized/model-based, not measured watts",
            "measured_optical_channel_claimed": False,
            "lut_axis": "waveform-sample SNR in dB; legacy column name ebn0_db",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lut_csv")
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/02_link/link_verification.json",
    )
    args = parser.parse_args()
    report = verify_lut(args.lut_csv)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
