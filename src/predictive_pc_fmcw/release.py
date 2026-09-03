from __future__ import annotations

from pathlib import Path
from typing import Any

STALE_MANUSCRIPT_MARKERS = (
    "reproducible research draft",
    "remain to be inserted before submission",
    "Fifty-one regression tests",
    "In 12 controlled episodes",
    "compact three-scene WOMD proxy",
    "full 1,125-row, five-seed staged study",
    "artifacts/corrected_v2/",
    "remain necessary for submission-quality empirical evidence",
)


def verify_release_readiness(
    manuscript_path: str | Path,
    required_evidence: list[str | Path],
) -> dict[str, Any]:
    manuscript = Path(manuscript_path)
    text = manuscript.read_text(encoding="utf-8")
    evidence = [Path(path) for path in required_evidence]
    stale = [marker for marker in STALE_MANUSCRIPT_MARKERS if marker in text]
    missing = [str(path) for path in evidence if not path.is_file()]
    checks = {
        "manuscript_exists": manuscript.is_file(),
        "no_stale_draft_markers": not stale,
        "all_required_official_evidence_exists": not missing,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "BLOCKED",
        "manuscript": str(manuscript),
        "checks": checks,
        "stale_markers": stale,
        "missing_evidence": missing,
        "required_evidence": [str(path) for path in evidence],
    }
