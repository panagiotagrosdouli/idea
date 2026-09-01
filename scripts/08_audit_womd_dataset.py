from __future__ import annotations

import argparse
import json

from predictive_pc_fmcw.data.dataset_audit import (
    audit_training_npz,
    write_audit_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a WOMD training NPZ and write reproducible statistics."
    )
    parser.add_argument("dataset")
    parser.add_argument("--output", default="artifacts/womd_dataset_audit.json")
    args = parser.parse_args()
    report = audit_training_npz(args.dataset)
    destination = write_audit_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
