#!/usr/bin/env python3
"""Build or validate the parallel Synthetic Dataset v1 protocol."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

from predictive_pc_fmcw.synthetic.configuration import (
    load_synthetic_protocol_config,
)
from predictive_pc_fmcw.synthetic.dataset import build_dataset, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/synthetic_dataset_v1")
    parser.add_argument(
        "--protocol-config",
        default="configs/synthetic_dataset_v1.json",
    )
    parser.add_argument("--scenarios-per-family", type=int, default=None)
    parser.add_argument("--ood-scenarios-per-family", type=int, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if args.validate_only:
        report = validate_dataset(output)
    else:
        if output.exists() and any(output.iterdir()):
            if not args.force:
                raise FileExistsError(
                    f"refusing to overwrite synthetic dataset directory: {output}"
                )
            shutil.rmtree(output)

        loaded = load_synthetic_protocol_config(args.protocol_config)
        config = loaded.build_config
        overrides: dict[str, int] = {}
        if args.scenarios_per_family is not None:
            overrides["scenarios_per_family"] = args.scenarios_per_family
        if args.ood_scenarios_per_family is not None:
            overrides["ood_scenarios_per_family"] = args.ood_scenarios_per_family
        if args.master_seed is not None:
            overrides["master_seed"] = args.master_seed
        if overrides:
            config = replace(config, **overrides)

        manifest = build_dataset(output, config=config)
        protocol_copy = output / "protocol_config.json"
        shutil.copyfile(args.protocol_config, protocol_copy)
        (output / "protocol_config.sha256").write_text(
            loaded.sha256 + "\n", encoding="utf-8"
        )
        report = validate_dataset(output)
        report["built_scenario_count"] = manifest["scenario_count"]
        report["protocol_config_sha256"] = loaded.sha256
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
