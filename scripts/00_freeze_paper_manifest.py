from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.data.manifest import write_compact_womd_manifest


def _git_sha(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "uncommitted"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--womd-export", default="data/example/womd_trajectories.json"
    )
    parser.add_argument("--output", default="artifacts/paper_manifest.json")
    parser.add_argument(
        "--womd-release", default="unknown-supplied-compact-export"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    dataset_manifest = output.with_name("womd_dataset_manifest.json")
    write_compact_womd_manifest(
        root / args.womd_export,
        dataset_manifest,
        release=args.womd_release,
    )
    report = {
        "git_commit": _git_sha(root),
        "config_path": args.config,
        "config": load_config(root / args.config).to_dict(),
        "dataset_manifest": str(dataset_manifest.relative_to(root)),
        "randomness": "All NumPy and training seeds are explicit in configs.",
        "scope": "real mobility + model-based PC-FMCW/DPSK communication",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
