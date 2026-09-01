from __future__ import annotations

import argparse

from predictive_pc_fmcw.data.training_export import (
    build_training_npz_from_scenarios,
)
from predictive_pc_fmcw.data.womd_official import load_official_womd_tfrecords


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build true-SDC training samples from official WOMD TFRecords."
    )
    parser.add_argument("tfrecords", nargs="+")
    parser.add_argument("--output", default="data/processed/womd_official_samples.npz")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--max-vehicles", type=int, default=16)
    parser.add_argument(
        "--fixed-split",
        help="Label every sample identically, e.g. official_validation.",
    )
    args = parser.parse_args()
    scenarios = load_official_womd_tfrecords(
        args.tfrecords,
        max_scenarios=args.max_scenarios,
        max_vehicles=args.max_vehicles,
    )
    output = build_training_npz_from_scenarios(
        scenarios,
        args.output,
        source="real_WOMD_v1.3.1_true_SDC_geometry",
        fixed_split=args.fixed_split,
    )
    print(output)


if __name__ == "__main__":
    main()
