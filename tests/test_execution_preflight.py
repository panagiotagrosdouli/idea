from __future__ import annotations

import tempfile
from pathlib import Path

from predictive_pc_fmcw.execution_preflight import canonical_execution_preflight


def test_preflight_passes_with_required_files() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "configs").mkdir()
        (root / "scripts").mkdir()
        link = root / "artifacts/paper_final/02_link"
        link.mkdir(parents=True)
        (root / "configs/womd_paper_corpus.json").write_text("{}", encoding="utf-8")
        (root / "scripts/run_canonical_womd_pipeline.py").write_text(
            "", encoding="utf-8"
        )
        (link / "dbpsk_ber_lut.csv").write_text("snr,ber\n0,0.1\n", encoding="utf-8")
        (link / "link_verification.json").write_text("{}", encoding="utf-8")
        data = root / "data"
        (data / "validation").mkdir(parents=True)
        train = data / "train.npz"
        validation = data / "validation.npz"
        train.write_bytes(b"npz")
        validation.write_bytes(b"npz")
        tfrecord = data / "validation/validation.tfrecord-00000-of-00150"
        tfrecord.write_bytes(b"record")

        report = canonical_execution_preflight(
            repo_root=root,
            data_root=data,
            train_npz=train,
            validation_npz=validation,
            validation_glob=str(data / "validation/*.tfrecord-*"),
        )
        assert report["status"] == "PASS"
        assert report["validation_tfrecord_count"] == 1


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
