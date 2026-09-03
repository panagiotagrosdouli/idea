import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.release import verify_release_readiness


class ReleaseReadinessTest(unittest.TestCase):
    def test_stale_draft_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manuscript = root / "paper.md"
            manuscript.write_text(
                "# Paper\n\n**Manuscript status:** reproducible research draft.\n",
                encoding="utf-8",
            )
            evidence = root / "official.json"
            evidence.write_text("{}", encoding="utf-8")
            report = verify_release_readiness(manuscript, [evidence])
            self.assertEqual(report["status"], "BLOCKED")
            self.assertFalse(report["checks"]["no_stale_draft_markers"])

    def test_clean_manuscript_with_evidence_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manuscript = root / "paper.md"
            manuscript.write_text(
                "# Evidence-backed final manuscript\n\nOfficial results are versioned.\n",
                encoding="utf-8",
            )
            evidence = root / "official.json"
            evidence.write_text("{}", encoding="utf-8")
            report = verify_release_readiness(manuscript, [evidence])
            self.assertEqual(report["status"], "PASS")

    def test_missing_evidence_is_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manuscript = root / "paper.md"
            manuscript.write_text("# Final manuscript\n", encoding="utf-8")
            report = verify_release_readiness(manuscript, [root / "missing.json"])
            self.assertEqual(report["status"], "BLOCKED")
            self.assertFalse(
                report["checks"]["all_required_official_evidence_exists"]
            )


if __name__ == "__main__":
    unittest.main()
