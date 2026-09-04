from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.data.dataset_audit import audit_training_npz

HISTORICAL_TRAINING = {
    "sample_count": 249137,
    "unique_scenarios": 24182,
    "history_steps": 11,
    "future_steps": 80,
    "sha256": "b47faf427487a7405531e4944c5bfff9ca56d4fcb9ce3f8495df3cce534347ee",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a rebuilt WOMD training NPZ against the archived historical "
            "paper-corpus fingerprint without forcing the old counts."
        )
    )
    parser.add_argument("training_npz")
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/01_data/historical_fingerprint.json",
    )
    parser.add_argument(
        "--require-exact",
        action="store_true",
        help="Exit nonzero unless all historical fingerprint fields match exactly.",
    )
    args = parser.parse_args()

    audit = audit_training_npz(args.training_npz)
    checks = {
        key: audit[key] == expected
        for key, expected in HISTORICAL_TRAINING.items()
    }
    exact_match = all(checks.values())
    report = {
        "status": "EXACT_MATCH" if exact_match else "DEVIATION_REQUIRES_EXPLANATION",
        "historical_reference": HISTORICAL_TRAINING,
        "rebuilt": {key: audit[key] for key in HISTORICAL_TRAINING},
        "checks": checks,
        "policy": (
            "Historical counts are a provenance fingerprint, not a target. "
            "Do not alter deterministic preprocessing merely to reproduce them; "
            "document any scientifically justified deviation."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_exact and not exact_match:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
