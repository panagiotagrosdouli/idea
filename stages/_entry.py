from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run(stage_id: str) -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + existing if existing else ""
    )
    command = [
        sys.executable,
        str(root / "scripts" / "run_research_stage.py"),
        "--stage",
        stage_id,
    ]
    if "--check" not in sys.argv[1:]:
        command.append("--execute")
    subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=True,
    )
