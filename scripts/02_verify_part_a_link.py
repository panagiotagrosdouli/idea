from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.link_verification import verify_lut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lut_csv")
    parser.add_argument("--chirp-diagnostic")
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/02_link/link_verification.json",
    )
    args = parser.parse_args()
    report = verify_lut(args.lut_csv, chirp_diagnostic_path=args.chirp_diagnostic)
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
