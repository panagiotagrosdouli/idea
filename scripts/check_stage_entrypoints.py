from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage_dirs = sorted(
        path for path in (root / "stages").iterdir() if path.is_dir()
    )
    run_files = [
        path / "run.py"
        for path in stage_dirs
        if (path / "stage.json").is_file()
    ]
    if not run_files:
        raise SystemExit("No executable stage entrypoints found.")
    for run_file in run_files:
        subprocess.run(
            [sys.executable, str(run_file), "--check"],
            cwd=root,
            check=True,
        )
    print(f"Checked {len(run_files)} stage entrypoints successfully.")


if __name__ == "__main__":
    main()
