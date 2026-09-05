from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy
import scipy
from matplotlib import __version__ as matplotlib_version

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "reproducibility_manifest.json"
INCLUDED_ROOTS = (
    "src",
    "tests",
    "scripts",
    "configs",
    "docs",
    "paper",
    "stages",
    "data/example",
    "reference/part_b_stage4",
)
EVIDENCE_ROOTS = tuple(
    f"artifacts/paper_final/{index:02d}_{name}"
    for index, name in (
        (0, "freeze"),
        (1, "data"),
        (2, "link"),
        (3, "baselines"),
        (4, "learning"),
        (5, "heldout"),
        (6, "scheduling"),
        (7, "analysis"),
    )
)
INCLUDED_FILES = (
    ".gitignore",
    ".github/workflows/ci.yml",
    "README.md",
    "README_GR.md",
    "Makefile",
    "pyproject.toml",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_roots(roots: tuple[str, ...]) -> list[dict[str, object]]:
    files = []
    for relative_root in roots:
        root = ROOT / relative_root
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if (
                "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or any(part.endswith(".egg-info") for part in path.parts)
            ):
                continue
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return files


def _run_text(command: list[str]) -> str | None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_git_state() -> dict[str, Any]:
    commit = _run_text(["git", "rev-parse", "HEAD"])
    status = _run_text(["git", "status", "--porcelain=v1"])
    diff = _run_text(["git", "diff", "--binary", "HEAD"])
    return {
        "commit": commit or "unavailable",
        "dirty": bool(status),
        "status_porcelain": status or "",
        "working_tree_diff_sha256": (
            hashlib.sha256((diff or "").encode("utf-8")).hexdigest()
        ),
    }


def collect_python_environment() -> dict[str, Any]:
    pip_freeze = _run_text([sys.executable, "-m", "pip", "freeze"])
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib_version,
        "pip_freeze": sorted((pip_freeze or "").splitlines()),
    }


def collect_torch_environment() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False}

    import torch

    cuda_available = bool(torch.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "available": True,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": len(devices),
        "devices": devices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash the executable research package and canonical evidence."
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT.relative_to(ROOT)),
        help="Manifest path, relative to the repository unless absolute.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    files = _inventory_roots(INCLUDED_ROOTS)
    for relative_path in INCLUDED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            continue
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    evidence = _inventory_roots(EVIDENCE_ROOTS)
    git_state = collect_git_state()
    report = {
        "status": "PASS" if not git_state["dirty"] else "DIRTY_WORKTREE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state,
        "environment": collect_python_environment(),
        "torch": collect_torch_environment(),
        "scientific_scope": {
            "mobility": "official WOMD required for final empirical claims",
            "communication": "model-based PC-FMCW/DPSK simulation",
            "measured_optical_channel": False,
            "official_validation_required_for_final_claims": True,
            "independent_statistical_unit": "WOMD scenario_id",
        },
        "source_file_count": len(files),
        "source_files": files,
        "evidence_file_count": len(evidence),
        "evidence_files": evidence,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
