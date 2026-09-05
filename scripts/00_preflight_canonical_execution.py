from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.execution_preflight import canonical_execution_preflight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for canonical WOMD paper execution."
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--validation-npz", required=True)
    parser.add_argument("--validation-glob")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=0.0)
    parser.add_argument(
        "--output",
        default="artifacts/paper_final/00_freeze/execution_preflight.json",
    )
    args = parser.parse_args()

    report = canonical_execution_preflight(
        repo_root=args.repo_root,
        data_root=args.data_root,
        train_npz=args.train_npz,
        validation_npz=args.validation_npz,
        validation_glob=args.validation_glob,
        full=args.full,
        require_gpu=args.require_gpu,
        min_free_gb=args.min_free_gb,
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
