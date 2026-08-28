from predictive_pc_fmcw.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "benchmark",
                "--output",
                "artifacts/reactive_schedulers",
                "--schedulers",
                "random",
                "round_robin",
                "reactive_greedy",
                "proportional_fair",
            ]
        )
    )
