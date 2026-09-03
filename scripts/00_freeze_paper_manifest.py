from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from predictive_pc_fmcw.config import load_config
from predictive_pc_fmcw.data.manifest import (
    audit_scenario_overlap,
    sha256_file,
    write_compact_womd_manifest,
)


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
    parser.add_argument("--womd-export", default="data/example/womd_trajectories.json")
    parser.add_argument("--output", default="artifacts/paper_manifest.json")
    parser.add_argument("--womd-release", default="unknown-supplied-compact-export")
    parser.add_argument("--training-npz")
    parser.add_argument("--validation-npz")
    parser.add_argument("--ber-lut")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = root / args.output
    dataset_manifest = output.with_name("womd_dataset_manifest.json")
    if args.training_npz:
        training_npz = Path(args.training_npz).resolve()
        dataset_report: dict[str, object] = {
            "dataset": "Waymo Open Motion Dataset",
            "release": args.womd_release,
            "training_npz": str(training_npz),
            "training_sha256": sha256_file(training_npz),
        }
        if args.validation_npz:
            validation_npz = Path(args.validation_npz).resolve()
            dataset_report["official_validation_npz"] = str(validation_npz)
            dataset_report["official_validation_sha256"] = sha256_file(validation_npz)
            dataset_report["split_integrity"] = audit_scenario_overlap(
                {"training_corpus": training_npz, "official_validation": validation_npz}
            )
        dataset_manifest.write_text(
            json.dumps(dataset_report, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_compact_womd_manifest(
            root / args.womd_export,
            dataset_manifest,
            release=args.womd_release,
        )
    report = {
        "git_commit": _git_sha(root),
        "config_path": args.config,
        "config_sha256": sha256_file(root / args.config),
        "config": load_config(root / args.config).to_dict(),
        "dataset_manifest": str(dataset_manifest.relative_to(root)),
        "randomness": "All NumPy and training seeds are explicit in configs.",
        "scope": "real mobility + model-based PC-FMCW/DPSK communication",
    }
    if args.ber_lut:
        report["ber_lut"] = {
            "path": args.ber_lut,
            "sha256": sha256_file(args.ber_lut),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
