from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchStage:
    stage_id: str
    title: str
    depends_on: tuple[str, ...]
    required_inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    verification_reports: tuple[str, ...]
    acceptance: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]


def load_research_stages(path: str | Path) -> tuple[ResearchStage, ...]:
    source = Path(path)
    if source.is_dir():
        stage_files = sorted(source.glob("*/stage.json"))
        if not stage_files:
            raise ValueError(f"No */stage.json definitions found under {source}.")
        items = [
            json.loads(stage_file.read_text(encoding="utf-8"))
            for stage_file in stage_files
        ]
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        items = payload["stages"]
    stages = tuple(
        ResearchStage(
            stage_id=str(item["id"]),
            title=str(item["title"]),
            depends_on=tuple(item.get("depends_on", [])),
            required_inputs=tuple(item.get("required_inputs", [])),
            outputs=tuple(item.get("outputs", [])),
            verification_reports=tuple(item.get("verification_reports", [])),
            acceptance=tuple(item.get("acceptance", [])),
            commands=tuple(tuple(command) for command in item.get("commands", [])),
        )
        for item in items
    )
    validate_stage_graph(stages)
    return stages


def validate_stage_graph(stages: tuple[ResearchStage, ...]) -> None:
    identifiers = [stage.stage_id for stage in stages]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Stage identifiers must be unique.")
    known: set[str] = set()
    for stage in stages:
        missing = set(stage.depends_on) - known
        if missing:
            raise ValueError(
                f"{stage.stage_id} has unknown or forward dependencies: "
                f"{sorted(missing)}"
            )
        known.add(stage.stage_id)


def _expand(value: str, environment: dict[str, str]) -> str:
    expanded = value
    for name, replacement in environment.items():
        expanded = expanded.replace(f"${{{name}}}", replacement)
    return expanded


def unresolved_variables(value: str) -> list[str]:
    import re

    return sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)))


def stage_status(
    stages: tuple[ResearchStage, ...],
    root: str | Path,
    environment: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    project_root = Path(root)
    env = dict(os.environ if environment is None else environment)
    status_by_id: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for stage in stages:
        expanded_inputs = [_expand(item, env) for item in stage.required_inputs]
        expanded_outputs = [_expand(item, env) for item in stage.outputs]
        expanded_reports = [
            _expand(item, env) for item in stage.verification_reports
        ]
        command_arguments = [
            _expand(argument, env) for command in stage.commands for argument in command
        ]
        unresolved = sorted(
            {
                variable
                for item in (*expanded_inputs, *expanded_outputs, *command_arguments)
                for variable in unresolved_variables(item)
            }
        )

        def exists(item: str) -> bool:
            candidate = (
                str(project_root / item) if not Path(item).is_absolute() else item
            )
            return (
                bool(glob.glob(candidate))
                if any(char in item for char in "*?[")
                else Path(candidate).exists()
            )

        missing_inputs = [
            item
            for item in expanded_inputs
            if not unresolved_variables(item) and not exists(item)
        ]
        missing_outputs = [
            item
            for item in expanded_outputs
            if not unresolved_variables(item) and not exists(item)
        ]
        failed_reports: list[dict[str, str]] = []
        for item in expanded_reports:
            if unresolved_variables(item) or not exists(item):
                continue
            report_path = Path(item)
            if not report_path.is_absolute():
                report_path = project_root / report_path
            try:
                report_status = str(
                    json.loads(report_path.read_text(encoding="utf-8")).get("status")
                )
            except (OSError, ValueError, AttributeError):
                report_status = "INVALID"
            if report_status != "PASS":
                failed_reports.append({"path": item, "status": report_status})
        blocked_dependencies = [
            dependency
            for dependency in stage.depends_on
            if status_by_id.get(dependency) != "complete"
        ]
        if failed_reports:
            state = "failed"
        elif not unresolved and not missing_outputs:
            state = "complete"
        elif unresolved or missing_inputs or blocked_dependencies:
            state = "blocked"
        else:
            state = "ready"
        status_by_id[stage.stage_id] = state
        rows.append(
            {
                "id": stage.stage_id,
                "title": stage.title,
                "status": state,
                "unresolved_variables": unresolved,
                "missing_inputs": missing_inputs,
                "missing_outputs": missing_outputs,
                "failed_verification_reports": failed_reports,
                "blocked_dependencies": blocked_dependencies,
                "acceptance": list(stage.acceptance),
            }
        )
    return rows


def expanded_commands(
    stage: ResearchStage, environment: dict[str, str] | None = None
) -> tuple[tuple[str, ...], ...]:
    env = dict(os.environ if environment is None else environment)
    commands = tuple(
        tuple(_expand(argument, env) for argument in command)
        for command in stage.commands
    )
    missing = sorted(
        {
            variable
            for command in commands
            for argument in command
            for variable in unresolved_variables(argument)
        }
    )
    if missing:
        raise ValueError(f"Missing stage environment variables: {missing}")
    return commands


def expand_command_globs(command: tuple[str, ...], root: str | Path) -> tuple[str, ...]:
    project_root = Path(root)
    expanded: list[str] = []
    for argument in command:
        if any(character in argument for character in "*?["):
            pattern = (
                argument
                if Path(argument).is_absolute()
                else str(project_root / argument)
            )
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise ValueError(f"Command glob matched no files: {argument}")
            expanded.extend(matches)
        else:
            expanded.append(argument)
    return tuple(expanded)
