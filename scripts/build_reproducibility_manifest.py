from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

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
    "data/example",
    "reference/part_b_stage4",
)
INCLUDED_FILES = (
    ".gitignore",
    ".github/workflows/ci.yml",
    "README.md",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements.lock",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hash the executable research package and record its environment."
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
    files = []
    for relative_root in INCLUDED_ROOTS:
        root = ROOT / relative_root
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
    for relative_path in INCLUDED_FILES:
        path = ROOT / relative_path
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    report = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": (
            git_result.stdout.strip()
            if git_result.returncode == 0
            else "uncommitted-workspace"
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib_version,
            "pytorch_available": importlib.util.find_spec("torch") is not None,
        },
        "scientific_scope": {
            "mobility": "controlled synthetic and compact real-WOMD export",
            "communication": "model-based PC-FMCW/DPSK simulation",
            "measured_optical_channel": False,
            "true_womd_ego_in_compact_export": False,
        },
        "file_count": len(files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
