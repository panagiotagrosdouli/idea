from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.data.manifest import audit_scenario_overlap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject scenario leakage across WOMD NPZ corpora."
    )
    parser.add_argument(
        "corpus",
        nargs="+",
        help="Named corpus in NAME=PATH form (for example train=training.npz).",
    )
    parser.add_argument("--output", default="artifacts/paper_final/split_audit.json")
    args = parser.parse_args()
    named: dict[str, Path] = {}
    for item in args.corpus:
        if "=" not in item:
            parser.error(f"Expected NAME=PATH, received {item!r}.")
        name, raw_path = item.split("=", 1)
        if not name or name in named:
            parser.error(f"Corpus name is empty or duplicated: {name!r}.")
        named[name] = Path(raw_path)
    report = audit_scenario_overlap(named)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("FAILED: scenario-level leakage was detected.")


if __name__ == "__main__":
    main()
