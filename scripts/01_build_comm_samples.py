from __future__ import annotations

import argparse

from predictive_pc_fmcw.data.training_export import (
    build_relative_motion_training_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--womd-export", default="data/example/womd_trajectories.json"
    )
    parser.add_argument("--output", default="data/cache/comm_samples.npz")
    parser.add_argument("--max-vehicles", type=int)
    args = parser.parse_args()
    print(
        build_relative_motion_training_npz(
            args.womd_export, args.output, max_vehicles=args.max_vehicles
        )
    )


if __name__ == "__main__":
    main()
