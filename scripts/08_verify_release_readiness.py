from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.release import verify_release_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Block final-paper release while stale draft claims or evidence gaps remain."
        )
    )
    parser.add_argument("--manuscript", default="paper/PAPER_DRAFT.md")
    parser.add_argument("--evidence", nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/08_release/release_readiness.json",
    )
    args = parser.parse_args()
    report = verify_release_readiness(args.manuscript, args.evidence)
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
