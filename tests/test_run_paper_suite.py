import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paper_suite import generate_cases
from piper_demo import MuJoCoTabletop, SCENE
from run_paper_suite import downward_ik_precondition, main, plan_positions, record_trial
from scene_factory import write_scene


class SuiteRunnerTests(unittest.TestCase):
    def test_fixed_task_filters_only_ik_unreachable_scenes(self):
        case = generate_cases(20260923)[0]
        tabletop = MuJoCoTabletop(SCENE)
        try:
            original, _, original_attempt = plan_positions(case)
            self.assertEqual(original_attempt, 0)
            self.assertTrue(downward_ik_precondition(case, original, tabletop.backend))
            positions, _, attempt = plan_positions(case, ik_backend=tabletop.backend)
            self.assertEqual(attempt, 0)
            self.assertTrue(downward_ik_precondition(case, positions, tabletop.backend))
        finally:
            tabletop.disconnect()

    def test_scripted_model_records_generated_program_and_physics(self):
        case = generate_cases(37)[0]
        positions, _, _ = plan_positions(case)

        class ScriptedCompletions:
            def create(self, **kwargs):
                source = case.params["source"]
                target = case.params["target"]
                content = f"put_first_on_second({source!r}, {target!r})"
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

        client = SimpleNamespace(chat=SimpleNamespace(completions=ScriptedCompletions()))
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "trial.xml"
            bodies = write_scene(positions, scene)
            result = record_trial(case, scene, bodies, client, "scripted", "paper-sim")
        self.assertEqual(len(result["generations"]), 1)
        self.assertIn("put_first_on_second", result["generations"][0]["generated_code"])
        self.assertIn(case.params["source"], result["initial_xyz"])
        self.assertIn(case.params["source"], result["final_xyz"])
        self.assertEqual(result["phase"], "completed" if result["error"] is None else "execution")
        self.assertIsInstance(result["task_success"], bool)

    def test_notebook_examples_remain_a_separate_suite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "notebook"
            arguments = ["run_paper_suite.py", "--suite", "code-demo", "--plan-only",
                         "--seed", "19", "--output-dir", str(output)]
            with patch("sys.argv", arguments):
                main()
            entries = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 5)
            self.assertTrue(all(entry["suite"] == "code-demo" and entry["protocol"] == "demo"
                                for entry in entries))

    def test_plan_creates_distinct_scenes_without_claiming_model_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paper"
            arguments = ["run_paper_suite.py", "--plan-only", "--seed", "8", "--trials", "2",
                         "--output-dir", str(output)]
            with patch("sys.argv", arguments):
                main()
            entries = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
            self.assertEqual(len(entries), 44)
            self.assertEqual(len(list((output / "scenes").glob("*.xml"))), 44)
            self.assertTrue(all(entry["phase"] == "planned" and entry["model"] is None for entry in entries))
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["case_count"], 44)
            self.assertTrue(summary["planned_only"])
            self.assertTrue(all(cell["successes"] == cell["failures"] == 0
                                for group in summary["groups"].values() for cell in group.values()))
            with patch("sys.argv", arguments), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
