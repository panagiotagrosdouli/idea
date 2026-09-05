from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_reproducibility_manifest.py"


def load_manifest_module():
    spec = importlib.util.spec_from_file_location("build_reproducibility_manifest", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reproducibility_manifest_captures_source_and_environment_state() -> None:
    module = load_manifest_module()
    git_state = module.collect_git_state()
    environment = module.collect_python_environment()
    torch_state = module.collect_torch_environment()

    assert len(git_state["commit"]) >= 7
    assert "source_dirty" in git_state
    assert len(git_state["source_diff_sha256"]) == 64
    assert environment["python"]
    assert environment["python_executable"]
    assert isinstance(environment["pip_freeze"], list)
    assert "available" in torch_state
