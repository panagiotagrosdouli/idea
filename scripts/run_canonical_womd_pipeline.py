from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

CANONICAL_SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")


def stage1(root: Path, data_root: Path, train_npz: Path, val_npz: Path) -> None:
    out = root / "artifacts/paper_final/01_data"
    out.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "scripts/01_materialize_womd_manifest.py",
            str(data_root),
            "--config",
            "configs/womd_paper_corpus.json",
            "--output",
            str(out / "source_shard_manifest.json"),
        ],
        cwd=root,
    )
    for source, name in (
        (train_npz, "training_audit.json"),
        (val_npz, "validation_audit.json"),
    ):
        run(
            [
                sys.executable,
                "scripts/08_audit_womd_dataset.py",
                str(source),
                "--output",
                str(out / name),
            ],
            cwd=root,
        )
    run(
        [
            sys.executable,
            "scripts/01_verify_historical_womd_fingerprint.py",
            str(train_npz),
            "--output",
            str(out / "historical_fingerprint.json"),
        ],
        cwd=root,
    )
    run(
        [
            sys.executable,
            "scripts/01_verify_womd_corpora.py",
            str(train_npz),
            str(val_npz),
            "--output",
            str(out / "corpus_verification.json"),
        ],
        cwd=root,
    )


def freeze_selection(
    root: Path,
    train_npz: Path,
    sweep: Path,
    selection: Path,
    lambda_link: float | None,
    lambda_outage: float | None,
    rationale: str,
) -> None:
    if selection.is_file():
        return
    if lambda_link is None or lambda_outage is None:
        raise RuntimeError(
            "No frozen lambda selection exists. Inspect the development-only sweep, "
            "then rerun with --lambda-link and --lambda-outage."
        )
    run(
        [
            sys.executable,
            "scripts/04_freeze_lambda_selection.py",
            str(sweep),
            str(train_npz),
            "--lambda-link",
            str(lambda_link),
            "--lambda-outage",
            str(lambda_outage),
            "--rationale",
            rationale,
            "--output",
            str(selection),
        ],
        cwd=root,
    )


def downstream(
    root: Path,
    train_npz: Path,
    val_npz: Path,
    validation_glob: str,
    lambda_link: float | None,
    lambda_outage: float | None,
    rationale: str,
) -> None:
    ber = root / "artifacts/paper_final/02_link/dbpsk_ber_lut.csv"
    require_file(ber, "Stage-2 BER LUT")
    seeds = [str(seed) for seed in CANONICAL_SEEDS]
    run(
        [
            sys.executable,
            "scripts/03_eval_npz_baselines.py",
            str(train_npz),
            "--split",
            "development",
            "--ber-lut",
            str(ber),
            "--output",
            "artifacts/paper_final/03_baselines",
        ],
        cwd=root,
    )
    sweep = root / "artifacts/paper_final/04_learning/lambda_sweep"
    run(
        [
            sys.executable,
            "scripts/04_run_lambda_sweep.py",
            str(train_npz),
            "--ber-lut",
            str(ber),
            "--output",
            str(sweep),
            "--epochs",
            "80",
            "--batch-size",
            "32",
            "--seeds",
            *seeds,
        ],
        cwd=root,
    )
    selection = root / "artifacts/paper_final/04_learning/lambda_selection.json"
    freeze_selection(
        root,
        train_npz,
        sweep,
        selection,
        lambda_link,
        lambda_outage,
        rationale,
    )
    learned = root / "artifacts/paper_final/04_learning/learned_ablation"
    run(
        [
            sys.executable,
            "scripts/04_run_training_ablation.py",
            str(train_npz),
            "--ber-lut",
            str(ber),
            "--output",
            str(learned),
            "--epochs",
            "80",
            "--batch-size",
            "32",
            "--seeds",
            *seeds,
            "--selection",
            str(selection),
        ],
        cwd=root,
    )
    checkpoints = sorted(learned.rglob("best_comm_aware_gru.pt"))
    if len(checkpoints) != 20:
        raise RuntimeError(f"Canonical archive incomplete: {len(checkpoints)}/20 checkpoints")
    completion = learned / "completion_manifest.json"
    require_file(completion, "Stage-4 completion manifest")
    heldout = root / "artifacts/paper_final/05_heldout"
    run(
        [
            sys.executable,
            "scripts/06_evaluate_learned_checkpoints.py",
            str(val_npz),
            *map(str, checkpoints),
            "--development-npz",
            str(train_npz),
            "--completion-manifest",
            str(completion),
            "--ber-lut",
            str(ber),
            "--output",
            str(heldout),
        ],
        cwd=root,
    )
    tfrecords = [Path(path) for path in sorted(glob.glob(validation_glob))]
    if not tfrecords:
        raise FileNotFoundError(f"No validation TFRecords matched: {validation_glob}")
    scheduling = root / "artifacts/paper_final/06_scheduling"
    run(
        [
            sys.executable,
            "scripts/06_evaluate_learned_scheduler_womd.py",
            *map(str, tfrecords),
            "--checkpoints",
            *map(str, checkpoints),
            "--training-npz",
            str(train_npz),
            "--completion-manifest",
            str(completion),
            "--ber-lut",
            str(ber),
            "--heldout-metrics",
            str(heldout / "heldout_metrics_by_scenario.csv"),
            "--output",
            str(scheduling),
        ],
        cwd=root,
    )
    run(
        [
            sys.executable,
            "scripts/07_analyze_learned_results.py",
            str(heldout / "heldout_metrics_by_scenario.csv"),
            "--output",
            "artifacts/paper_final/07_analysis/learned",
        ],
        cwd=root,
    )
    run(
        [
            sys.executable,
            "scripts/07_analyze_scheduler_utility.py",
            str(heldout / "heldout_metrics_by_scenario.csv"),
            str(scheduling),
            "--output",
            "artifacts/paper_final/07_analysis/statistics",
        ],
        cwd=root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical paper-scale WOMD runner.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--validation-npz", required=True)
    parser.add_argument("--validation-glob")
    parser.add_argument("--lambda-link", type=float)
    parser.add_argument("--lambda-outage", type=float)
    parser.add_argument(
        "--selection-rationale",
        default="Selected from the declared development-only sweep.",
    )
    parser.add_argument(
        "--mode",
        choices=("stage1", "full"),
        default="stage1",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    train_npz = Path(args.train_npz).resolve()
    val_npz = Path(args.validation_npz).resolve()
    data_root = Path(args.data_root).resolve()
    require_file(train_npz, "Training NPZ")
    require_file(val_npz, "Validation NPZ")
    stage1(root, data_root, train_npz, val_npz)
    if args.mode == "full":
        if not args.validation_glob:
            raise ValueError("--validation-glob is required for --mode full")
        downstream(
            root,
            train_npz,
            val_npz,
            args.validation_glob,
            args.lambda_link,
            args.lambda_outage,
            args.selection_rationale,
        )

    summary = {
        "mode": args.mode,
        "repo_root": str(root),
        "train_npz": str(train_npz),
        "validation_npz": str(val_npz),
        "canonical_seeds": CANONICAL_SEEDS,
        "status": "PASS",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
