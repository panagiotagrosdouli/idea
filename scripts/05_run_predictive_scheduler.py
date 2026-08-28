from predictive_pc_fmcw.cli import main


if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "benchmark",
                "--output",
                "artifacts/predictive_schedulers",
                "--schedulers",
                "cv_predictive",
                "kalman_predictive",
                "imm_predictive",
                "predictive_utility",
                "oracle",
            ]
        )
    )
