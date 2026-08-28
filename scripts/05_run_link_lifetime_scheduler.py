from predictive_pc_fmcw.cli import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "benchmark",
                "--output",
                "artifacts/link_lifetime_scheduler",
                "--schedulers",
                "reactive_greedy",
                "predictive_utility",
                "link_lifetime",
                "oracle",
            ]
        )
    )
