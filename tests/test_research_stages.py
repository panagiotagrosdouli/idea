import json
import tempfile
import unittest
from pathlib import Path

from predictive_pc_fmcw.research_stages import (
    expanded_commands,
    load_research_stages,
    stage_status,
)


class ResearchStagesTest(unittest.TestCase):
    def test_repository_stage_graph_is_valid_and_ordered(self):
        stages = load_research_stages("stages")
        self.assertEqual(
            [stage.stage_id for stage in stages], [f"stage{i}" for i in range(9)]
        )
        self.assertEqual(stages[-1].depends_on, ("stage7",))

    def test_canonical_stage1_and_stage6_contracts_are_frozen(self):
        stages = {stage.stage_id: stage for stage in load_research_stages("stages")}
        stage1 = stages["stage1"]
        self.assertIn("${WOMD_DATA_ROOT}", stage1.required_inputs)
        self.assertIn(
            "artifacts/paper_final/01_data/source_shard_manifest.json",
            stage1.outputs,
        )
        self.assertIn(
            "artifacts/paper_final/01_data/historical_fingerprint.json",
            stage1.outputs,
        )

        stage6 = stages["stage6"]
        flattened = [token for command in stage6.commands for token in command]
        self.assertIn("--canonical", flattened)
        self.assertIn("exactly five paired traffic seeds", stage6.acceptance)
        self.assertIn("exactly eight frozen scheduler families", stage6.acceptance)

    def test_stage8_requires_end_to_end_provenance(self):
        stages = {stage.stage_id: stage for stage in load_research_stages("stages")}
        stage8 = stages["stage8"]
        required = set(stage8.required_inputs)
        self.assertIn(
            "artifacts/paper_final/01_data/source_shard_manifest.json", required
        )
        self.assertIn(
            "artifacts/paper_final/04_learning/learned_ablation/completion_manifest.json",
            required,
        )
        self.assertIn(
            "artifacts/paper_final/07_analysis/learned/"
            "heldout_link_fidelity_by_scenario.csv",
            required,
        )
        self.assertIn(
            "artifacts/paper_final/07_analysis/statistics/"
            "ade_vs_scheduler_utility_by_scenario.csv",
            required,
        )

    def test_status_blocks_unresolved_inputs_and_unlocks_in_order(self):
        payload = {
            "stages": [
                {"id": "stage0", "title": "zero", "outputs": ["a.json"]},
                {
                    "id": "stage1",
                    "title": "one",
                    "depends_on": ["stage0"],
                    "required_inputs": ["${DATA}"],
                    "outputs": ["b.json"],
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "stages.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            stages = load_research_stages(config)
            rows = stage_status(stages, root, environment={})
            self.assertEqual(rows[0]["status"], "ready")
            self.assertEqual(rows[1]["status"], "blocked")
            (root / "a.json").write_text("{}", encoding="utf-8")
            (root / "data.npz").write_text("x", encoding="utf-8")
            rows = stage_status(stages, root, environment={"DATA": "data.npz"})
            self.assertEqual(rows[0]["status"], "complete")
            self.assertEqual(rows[1]["status"], "ready")

    def test_command_expansion_is_fail_closed(self):
        stage = load_research_stages("stages")[0]
        with self.assertRaises(ValueError):
            expanded_commands(stage, environment={})

    def test_failed_verification_report_prevents_complete_status(self):
        payload = {
            "stages": [
                {
                    "id": "stage0",
                    "title": "verified",
                    "outputs": ["artifact.csv", "verification.json"],
                    "verification_reports": ["verification.json"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "stages.json"
            config.write_text(json.dumps(payload), encoding="utf-8")
            (root / "artifact.csv").write_text("value\n1\n", encoding="utf-8")
            report = root / "verification.json"
            report.write_text('{"status": "FAIL"}', encoding="utf-8")
            stages = load_research_stages(config)
            rows = stage_status(stages, root, environment={})
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(
                rows[0]["failed_verification_reports"],
                [{"path": "verification.json", "status": "FAIL"}],
            )
            report.write_text('{"status": "PASS"}', encoding="utf-8")
            self.assertEqual(
                stage_status(stages, root, environment={})[0]["status"], "complete"
            )


if __name__ == "__main__":
    unittest.main()
