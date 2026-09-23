import unittest

import numpy as np

from paper_suite import SEEN_SEEN, UNSEEN_UNSEEN, instantiate
from code_suite import generate_code_cases
from run_paper_suite import plan_positions
from scene_factory import sample_positions
from task_scoring import chosen_block, corner_positions, near, score_case, stacked


def xyz_scene(case):
    positions = sample_positions(case.objects, case.seed)
    return {name: np.array([*xy, 0.772 if name.endswith(" block") else 0.75])
            for name, xy in positions.items()}


class TaskScoringTests(unittest.TestCase):
    def test_notebook_cases_score_against_initial_referents(self):
        cases = generate_code_cases(43)
        self.assertEqual(len(cases), 5)
        self.assertEqual(cases, generate_code_cases(43))
        farthest_case = next(case for case in cases if case.task_type == "demo_farthest")
        initial = xyz_scene(farthest_case)
        final = {name: value.copy() for name, value in initial.items()}
        bowls = farthest_case.goal["bowls"]
        farthest = max(bowls, key=lambda name: np.linalg.norm(initial[name][:2] - initial["red block"][:2]))
        final["red block"][:2] = initial[farthest][:2]
        self.assertTrue(score_case(farthest_case, initial, final))

    def test_single_stack_requires_height_not_just_xy(self):
        case = next(instantiate("seen_1", SEEN_SEEN, seed) for seed in range(100)
                    if instantiate("seen_1", SEEN_SEEN, seed).params["target"].endswith(" block"))
        initial = xyz_scene(case)
        final = {name: value.copy() for name, value in initial.items()}
        source, target = case.params["source"], case.params["target"]
        final[source][:2] = initial[target][:2]
        self.assertFalse(score_case(case, initial, final))
        final[source][2] = 0.82
        self.assertTrue(score_case(case, initial, final))

    def test_nearest_referent_is_fixed_by_initial_state(self):
        case = instantiate("seen_7", SEEN_SEEN, 7)
        initial = xyz_scene(case)
        chosen = chosen_block(case, initial)
        other = next(name for name in case.goal["blocks"] if name != chosen)
        final = {name: value.copy() for name, value in initial.items()}
        final[other][:2] = initial[case.params["bowl"]][:2]
        self.assertEqual(chosen_block(case, initial), chosen)

    def test_mismatched_bowls_require_distinct_nonmatching_targets(self):
        case = instantiate("unseen_2", UNSEEN_UNSEEN, 8)
        initial = xyz_scene(case)
        final = {name: value.copy() for name, value in initial.items()}
        for name in case.goal["blocks"]:
            final[name][:2] = initial[f"{name.split()[0]} bowl"][:2]
        self.assertFalse(score_case(case, initial, final))
        for index, name in enumerate(case.goal["blocks"]):
            bowl = case.goal["bowls"][(index + 1) % len(case.goal["bowls"])]
            if bowl.split()[0] == name.split()[0]:
                bowl = case.goal["bowls"][(index + 2) % len(case.goal["bowls"])]
            final[name][:2] = initial[bowl][:2]
        self.assertTrue(score_case(case, initial, final))

    def test_line_and_stack_require_geometric_success(self):
        case = next(instantiate("unseen_6", UNSEEN_UNSEEN, seed) for seed in range(100)
                    if instantiate("unseen_6", UNSEEN_UNSEEN, seed).params["line"] == "diagonal")
        initial = xyz_scene(case)
        final = {name: value.copy() for name, value in initial.items()}
        for name, xy in zip(case.goal["blocks"], [(-0.15, -0.15), (-0.08, -0.08), (-0.01, -0.01)]):
            final[name][:2] = xy
        self.assertTrue(score_case(case, initial, final))
        final[case.goal["blocks"][0]][1] += 0.08
        self.assertFalse(score_case(case, initial, final))
        self.assertFalse(stacked(case.goal["blocks"], final))

    def test_every_generated_case_can_sample_a_valid_scene(self):
        from paper_suite import generate_cases
        from task_scoring import scene_precondition

        for case in generate_cases(71, repetitions=4):
            positions, scene_seed, attempts = plan_positions(case)
            self.assertTrue(scene_precondition(case, positions))
            self.assertLess(attempts, 128)
            self.assertEqual(positions, sample_positions(case.objects, scene_seed))


if __name__ == "__main__":
    unittest.main()
