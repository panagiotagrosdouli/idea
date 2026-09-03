from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from predictive_pc_fmcw.research_stages import (
    expand_command_globs,
    expanded_commands,
    load_research_stages,
    stage_status,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or execute the gated paper workflow one stage at a time."
    )
    parser.add_argument("--config", default="configs/research_stages.json")
    parser.add_argument("--stage", help="Stage identifier, for example stage0.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status-json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    stages = load_research_stages(root / args.config)
    rows = stage_status(stages, root)
    if args.status_json:
        destination = root / args.status_json
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if not args.stage:
        print(json.dumps(rows, indent=2))
        return
    selected = next((stage for stage in stages if stage.stage_id == args.stage), None)
    if selected is None:
        parser.error(f"Unknown stage {args.stage!r}.")
    current = next(row for row in rows if row["id"] == args.stage)
    print(json.dumps(current, indent=2))
    try:
        commands = expanded_commands(selected)
    except ValueError:
        if args.execute:
            raise
        commands = selected.commands
    print("Commands:")
    for command in commands:
        print(" ", " ".join(command))
    if not args.execute:
        return
    if current["blocked_dependencies"] or current["missing_inputs"]:
        raise SystemExit("Stage is blocked; close its dependencies and inputs first.")
    for command in commands:
        command = expand_command_globs(command, root)
        resolved = (sys.executable, *command[1:]) if command[0] == "python" else command
        subprocess.run(resolved, cwd=root, check=True)


if __name__ == "__main__":
    main()
