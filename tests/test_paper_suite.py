import json
import random
import unittest
from dataclasses import fields, replace

from paper_suite import (
    GROUPS,
    SEEN_ATTRIBUTES,
    SEEN_SEEN,
    SEEN_UNSEEN,
    UNSEEN_ATTRIBUTES,
    UNSEEN_UNSEEN,
    build_manifest,
    Case,
    generate_cases,
    instantiate,
)


class PaperSuiteTests(unittest.TestCase):
    def test_corner_attribute_is_named_as_corner_not_side(self):
        case = next(instantiate("seen_3", SEEN_SEEN, seed) for seed in range(100)
                    if "-" in instantiate("seen_3", SEEN_SEEN, seed).params["side"])
        self.assertIn(" corner.", case.command)
        self.assertNotIn("corner side", case.command)

    def test_exactly_22_independent_category_split_cells(self):
        self.assertEqual({split: len(categories) for split, categories in GROUPS.items()}, {
            SEEN_SEEN: 8, SEEN_UNSEEN: 8, UNSEEN_UNSEEN: 6,
        })
        cases = generate_cases(123, repetitions=3)
        self.assertEqual(len(cases), 66)
        self.assertEqual(len({case.id for case in cases}), 66)
        self.assertEqual(len({case.seed for case in cases}), 66)
        for split, categories in GROUPS.items():
            for category in categories:
                self.assertEqual(sum(case.split == split and case.category == category for case in cases), 3)

    def test_reproducible_order_independent_and_global_rng_unchanged(self):
        random.seed(875)
        before = random.getstate()
        cases = generate_cases(-1234, 4)
        self.assertEqual(before, random.getstate())
        self.assertEqual(cases, generate_cases(-1234, 4))
        self.assertNotEqual(cases, generate_cases(-1235, 4))
        self.assertEqual(cases[11], replace(
            instantiate(cases[11].category, cases[11].split, cases[11].seed), id=cases[11].id,
        ))

    def test_interface_and_manifest_are_json_serializable(self):
        self.assertEqual(tuple(field.name for field in fields(Case)), (
            "id", "split", "command", "objects", "seed", "task_type", "params", "goal", "metadata",
        ))
        cases = generate_cases(7)
        for case in cases:
            self.assertIsInstance(case.objects, tuple)
            self.assertEqual(case.task_type, case.category)
            self.assertEqual(case.params, case.goal["parameters"])
            self.assertEqual(case.metadata["category_name"], case.goal["kind"])
            self.assertEqual(set(case.expected["parameters"]), set(case.parameters))
            self.assertEqual(case.category_name, case.expected["kind"])
            self.assertTrue(case.command.endswith("."))
            self.assertEqual(json.loads(json.dumps(case.to_manifest()))["objects"], list(case.objects))
        manifest = build_manifest(7, 2)
        self.assertEqual(json.loads(json.dumps(manifest)), manifest)
        self.assertEqual(sum(len(entries) for group in manifest["groups"].values()
                             for entries in group.values()), 44)
        self.assertEqual(manifest["groups"][SEEN_SEEN]["seen_1"][0],
                         generate_cases(7, 2)[0].to_manifest())
        self.assertIn("not the original", manifest["scope"])

    def test_color_and_attribute_pool_boundaries(self):
        for case in generate_cases(89, 30):
            attrs = SEEN_ATTRIBUTES if case.split == SEEN_SEEN else UNSEEN_ATTRIBUTES
            colors = attrs["colors"]
            self.assertTrue(all(name.split()[0] in colors for name in case.objects))
            self.assertEqual(len(case.objects), len(set(case.objects)))
            self.assertEqual(sum(name.endswith(" bowl") for name in case.objects),
                             1 if case.parameters.get("ordinal") == "fourth" else 3)
            self.assertEqual(sum(name.endswith(" block") for name in case.objects),
                             4 if case.parameters.get("ordinal") == "fourth" else 3)
            for key, pool in (("side", "sides"), ("direction", "directions"),
                              ("distance", "distances"), ("magnitude", "magnitudes"),
                              ("ordinal", "ordinals"), ("line", "lines")):
                if key in case.parameters:
                    self.assertIn(case.parameters[key], attrs[pool])

    def test_referents_and_structured_goals(self):
        for case in generate_cases(101, 16):
            params = case.parameters
            category = case.category
            if category == "seen_1":
                self.assertIn(params["source"], case.objects)
                self.assertIn(params["target"], case.objects)
                self.assertNotEqual(params["source"], params["target"])
            elif category == "seen_5":
                self.assertEqual({name.split()[0] for name in case.expected["blocks"]},
                                 {name.split()[0] for name in case.expected["bowls"]})
                self.assertEqual(case.expected["constraint"], "match_block_and_bowl_colors")
            elif category == "unseen_2":
                self.assertEqual({name.split()[0] for name in case.expected["blocks"]},
                                 {name.split()[0] for name in case.expected["bowls"]})
                self.assertEqual(case.expected["constraint"], "each_block_in_a_different_color_bowl")
                self.assertNotIn("targets", params)
            elif category == "unseen_1":
                self.assertEqual(case.expected["constraint"], "distinct_corners")
                self.assertNotIn("corners", params)
            elif category == "seen_6":
                self.assertEqual(case.expected["precondition"],
                                 "at_least_one_block_in_specified_direction_of_bowl")
            elif category == "unseen_5":
                self.assertTrue(params["block"].endswith(" block"))
                self.assertTrue(params["bowl"].endswith(" bowl"))
                self.assertIn(params["block"], case.command)
            if category in ("seen_6", "seen_7", "unseen_4", "unseen_5"):
                self.assertIn(params["bowl"], case.objects)

    def test_fourth_rank_needs_four_blocks_and_one_bowl(self):
        fourth = next(instantiate("seen_8", SEEN_UNSEEN, seed) for seed in range(100)
                      if instantiate("seen_8", SEEN_UNSEEN, seed).parameters["ordinal"] == "fourth")
        self.assertEqual(len(fourth.objects), 5)
        self.assertEqual(len(fourth.expected["blocks"]), 4)
        self.assertEqual(len(fourth.expected["bowls"]), 1)
        self.assertIn("fourth", fourth.command)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            instantiate("unseen_1", SEEN_SEEN, 1)
        with self.assertRaises(ValueError):
            instantiate("seen_1", "not_a_split", 1)
        for seed in (True, 1.5, "5"):
            with self.subTest(seed=seed), self.assertRaises(TypeError):
                generate_cases(seed)
        for repetitions in (0, -1, True, 1.5):
            with self.subTest(repetitions=repetitions), self.assertRaises(ValueError):
                generate_cases(0, repetitions)


if __name__ == "__main__":
    unittest.main()
