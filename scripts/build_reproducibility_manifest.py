from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy
import scipy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "reproducibility_manifest.json"
INCLUDED_ROOTS = ("src", "tests", "scripts", "configs", "docs", "data/example")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for relative_root in INCLUDED_ROOTS:
        root = ROOT / relative_root
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    report = {
        "status": "PASS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
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
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()

