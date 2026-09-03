from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.data.corpus_audit import verify_corpora


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of frozen WOMD training and validation corpora."
    )
    parser.add_argument("training_npz")
    parser.add_argument("validation_npz")
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/01_data/corpus_verification.json",
    )
    args = parser.parse_args()

    report = verify_corpora(args.training_npz, args.validation_npz)
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
