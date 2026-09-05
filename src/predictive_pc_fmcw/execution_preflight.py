from __future__ import annotations

import glob
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any


def canonical_execution_preflight(
    *,
    repo_root: str | Path,
    data_root: str | Path,
    train_npz: str | Path,
    validation_npz: str | Path,
    validation_glob: str | None = None,
    full: bool = False,
    require_gpu: bool = False,
    min_free_gb: float = 0.0,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    data = Path(data_root).resolve()
    train = Path(train_npz).resolve()
    validation = Path(validation_npz).resolve()

    required_repo_files = [
        root / "configs/womd_paper_corpus.json",
        root / "scripts/run_canonical_womd_pipeline.py",
    ]
    if full:
        required_repo_files.extend(
            [
                root / "artifacts/paper_final/02_link/dbpsk_ber_lut.csv",
                root / "artifacts/paper_final/02_link/link_verification.json",
            ]
        )
    validation_files = sorted(glob.glob(validation_glob)) if validation_glob else []
    free_gb = shutil.disk_usage(root).free / (1024**3)
    torch_available = importlib.util.find_spec("torch") is not None
    cuda_available = False
    if require_gpu and torch_available:
        import torch

        cuda_available = bool(torch.cuda.is_available())

    checks = {
        "python_supported": sys.version_info >= (3, 10),
        "repo_contract_files_present": all(path.is_file() for path in required_repo_files),
        "data_root_present": data.is_dir(),
        "training_npz_present": train.is_file(),
        "validation_npz_present": validation.is_file(),
        "validation_tfrecords_present": (
            bool(validation_files) if full and validation_glob else not full
        ),
        "torch_available": torch_available if require_gpu else True,
        "cuda_available": cuda_available if require_gpu else True,
        "free_disk_sufficient": free_gb >= min_free_gb,
    }
    missing_repo_files = [str(path) for path in required_repo_files if not path.is_file()]
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "mode": "full" if full else "stage1",
        "checks": checks,
        "repo_root": str(root),
        "data_root": str(data),
        "training_npz": str(train),
        "validation_npz": str(validation),
        "validation_glob": validation_glob,
        "validation_tfrecord_count": len(validation_files),
        "missing_repo_files": missing_repo_files,
        "python": sys.version.split()[0],
        "free_disk_gb": free_gb,
        "require_gpu": require_gpu,
        "torch_available_raw": torch_available,
        "cuda_available_raw": cuda_available,
    }
