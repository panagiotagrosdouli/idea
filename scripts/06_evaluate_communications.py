from predictive_pc_fmcw.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "benchmark",
                "--config",
                "configs/default.json",
                "--output",
                "artifacts/final_communication_evaluation",
            ]
        )
    )
