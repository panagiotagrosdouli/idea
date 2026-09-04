from __future__ import annotations

import argparse
import json
from pathlib import Path

from predictive_pc_fmcw.data.womd_preflight import preflight_womd_roots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed discovery and verification of Stage-1 WOMD inputs."
    )
    parser.add_argument("roots", nargs="+", help="Files or directories to inspect")
    parser.add_argument("--output", type=Path, default=Path("womd_preflight.json"))
    args = parser.parse_args()

    report = preflight_womd_roots(args.roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["stage1_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
