from __future__ import annotations

import json
import tempfile
from pathlib import Path

from predictive_pc_fmcw.execution_preflight import canonical_execution_preflight
from predictive_pc_fmcw.link_verification import verify_lut


def _base_tree(root: Path) -> tuple[Path, Path, Path]:
    (root / "configs").mkdir()
    (root / "scripts").mkdir()
    (root / "configs/womd_paper_corpus.json").write_text("{}", encoding="utf-8")
    (root / "scripts/run_canonical_womd_pipeline.py").write_text("", encoding="utf-8")
    data = root / "data"
    (data / "validation").mkdir(parents=True)
    train = data / "train.npz"
    validation = data / "validation.npz"
    train.write_bytes(b"npz")
    validation.write_bytes(b"npz")
    return data, train, validation


def _write_valid_stage2_fixture(link: Path) -> None:
    lut = link / "dbpsk_ber_lut.csv"
    header = (
        "ebn0_db,simulated_ber,bits,errors,ber_upper_95,ber_for_lut,"
        "receiver,snr_semantics\n"
    )
    rows = []
    for snr in range(-5, 26):
        ber = max(0.001, 0.3 - 0.009 * (snr + 5))
        rows.append(
            f"{snr},{ber:.6f},250000,1,{ber:.6f},{ber:.6f},"
            "supplied_part_a_fft_dpsk,waveform_sample_snr_db\n"
        )
    lut.write_text(header + "".join(rows), encoding="utf-8")
    verification = verify_lut(lut)
    assert verification["status"] == "PASS"
    (link / "link_verification.json").write_text(
        json.dumps(
            {
                "status": verification["status"],
                "sha256": verification["sha256"],
            }
        ),
        encoding="utf-8",
    )


def test_stage1_preflight_does_not_require_link_artifacts() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data, train, validation = _base_tree(root)
        report = canonical_execution_preflight(
            repo_root=root,
            data_root=data,
            train_npz=train,
            validation_npz=validation,
        )
        assert report["status"] == "PASS"
        assert report["mode"] == "stage1"


def test_full_preflight_passes_with_downstream_inputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data, train, validation = _base_tree(root)
        link = root / "artifacts/paper_final/02_link"
        link.mkdir(parents=True)
        _write_valid_stage2_fixture(link)
        tfrecord = data / "validation/validation.tfrecord-00000-of-00150"
        tfrecord.write_bytes(b"record")

        report = canonical_execution_preflight(
            repo_root=root,
            data_root=data,
            train_npz=train,
            validation_npz=validation,
            validation_glob=str(data / "validation/*.tfrecord-*"),
            full=True,
        )
        assert report["status"] == "PASS"
        assert report["mode"] == "full"
        assert report["validation_tfrecord_count"] == 1
        assert report["checks"]["frozen_stage2_valid"] is True


def test_preflight_blocks_missing_inputs() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report = canonical_execution_preflight(
            repo_root=root,
            data_root=root / "missing-data",
            train_npz=root / "missing-train.npz",
            validation_npz=root / "missing-validation.npz",
        )
        assert report["status"] == "BLOCKED"
        assert report["checks"]["repo_contract_files_present"] is False
        assert report["checks"]["training_npz_present"] is False
        assert report["checks"]["validation_npz_present"] is False
