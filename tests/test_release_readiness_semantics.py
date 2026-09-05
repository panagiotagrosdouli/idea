from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.release import verify_release_readiness

OBJECTIVES = [
    "trajectory_only",
    "trajectory_link",
    "trajectory_outage",
    "full",
]
SEEDS = [20260827, 20260828, 20260829, 20260830, 20260831]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class ReleaseReadinessSemanticsTest(unittest.TestCase):
    def test_semantic_evidence_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manuscript = root / "paper.md"
            manuscript.write_text(
                "Canonical evidence-backed manuscript.", encoding="utf-8"
            )

            source = root / "source_shard_manifest.json"
            corpus = root / "corpus_verification.json"
            link = root / "link_verification.json"
            completion = root / "completion_manifest.json"
            heldout = root / "heldout_provenance.json"
            scheduling = root / "evaluation_manifest.json"
            learned = root / "learned_heldout_analysis.json"
            utility = root / "scheduler_utility_summary.json"
            scenario_csv = root / "heldout_link_fidelity_by_scenario.csv"

            write_json(
                source,
                {
                    "complete": True,
                    "training": {"complete": True},
                    "validation": {"complete": True},
                },
            )
            write_json(corpus, {"status": "PASS"})
            write_json(link, {"status": "PASS"})
            write_json(
                completion,
                {
                    "complete": True,
                    "expected_runs": 20,
                    "completed_runs": 20,
                    "objectives": OBJECTIVES,
                    "seeds": SEEDS,
                    "checkpoints": [f"checkpoint-{index}" for index in range(20)],
                    "dataset_sha256": "dataset-hash",
                    "link_config_sha256": "link-hash",
                },
            )
            verified = {"status": "PASS"}
            write_json(
                heldout,
                {
                    "completion_verification": verified,
                    "ber_lut_verification": verified,
                    "link_model": {"ber_source": "lut"},
                },
            )
            write_json(
                scheduling,
                {
                    "canonical": True,
                    "scenario_count": 3,
                    "heldout_scenario_count": 3,
                    "checkpoints": [f"checkpoint-{index}" for index in range(20)],
                    "paired_schedulers": [f"scheduler-{index}" for index in range(8)],
                    "traffic_seeds": SEEDS,
                    "independent_statistical_unit": "scenario_id",
                    "completion_verification": verified,
                    "ber_lut_verification": verified,
                },
            )
            write_json(
                learned,
                {
                    "accuracy_link_correlations": {"independent_scenarios": 3},
                    "full_vs_trajectory_only": {"pairs": 3},
                    "objective_summary": {name: {} for name in OBJECTIVES},
                },
            )
            write_json(
                utility,
                {
                    **{name: {} for name in OBJECTIVES},
                    "ade_vs_realized_goodput_gain": {
                        "independent_scenarios": 3
                    },
                },
            )
            scenario_csv.write_text("scenario_id,ade_m\ns1,1.0\n", encoding="utf-8")

            evidence = [
                source,
                corpus,
                link,
                completion,
                heldout,
                scheduling,
                learned,
                utility,
                scenario_csv,
            ]
            passed = verify_release_readiness(manuscript, evidence)
            self.assertEqual(passed["status"], "PASS")
            self.assertEqual(passed["invalid_evidence"], [])

            write_json(corpus, {"status": "FAIL"})
            blocked = verify_release_readiness(manuscript, evidence)
            self.assertEqual(blocked["status"], "BLOCKED")
            self.assertEqual(
                blocked["invalid_evidence"],
                [
                    {
                        "path": str(corpus),
                        "reason": "corpus_verification.json status is not PASS",
                    }
                ],
            )


if __name__ == "__main__":
    unittest.main()
