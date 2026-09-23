import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from piper_demo import MuJoCoTabletop
from scene_factory import (FORWARD_OFFSET, FORWARD_OFFSET_METERS, WORKSPACE_MAX,
                           WORKSPACE_MIN, sample_positions, write_scene)


class SceneFactoryTests(unittest.TestCase):
    def test_objects_stay_in_the_downward_ik_workspace(self):
        names = ("yellow block", "orange block", "red block", "yellow bowl", "orange bowl", "blue bowl")
        seed = 20260923
        positions = sample_positions(names, seed)
        self.assertEqual(FORWARD_OFFSET_METERS, 0.0)
        np.testing.assert_allclose(FORWARD_OFFSET, [0.0, 0.0])
        np.testing.assert_allclose(WORKSPACE_MIN, [-0.19, -0.19])
        np.testing.assert_allclose(WORKSPACE_MAX, [0.01, 0.19])
        for position in positions.values():
            self.assertTrue(np.all(position >= WORKSPACE_MIN))
            self.assertTrue(np.all(position <= WORKSPACE_MAX))
            self.assertGreaterEqual(position[0] - (-0.3), 0.11)
            self.assertLessEqual(position[0] - (-0.3), 0.31)

    def test_seeded_scenes_are_reproducible_and_separated(self):
        names = ("cyan block", "purple block", "orange block", "red bowl", "blue bowl", "gray bowl")
        first = sample_positions(names, seed=42)
        self.assertEqual(first, sample_positions(names, seed=42))
        self.assertNotEqual(first, sample_positions(names, seed=43))
        for index, first_name in enumerate(names):
            for second_name in names[index + 1:]:
                self.assertGreaterEqual(np.linalg.norm(np.array(first[first_name]) - first[second_name]), 0.15)

    def test_generated_scene_loads_in_piper_mujoco(self):
        names = ("cyan block", "purple block", "orange block", "red bowl", "blue bowl", "gray bowl")
        positions = sample_positions(names, seed=42)
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "trial.xml"
            bodies = write_scene(positions, scene)
            root = ET.parse(scene).getroot()
            self.assertEqual(len(root.findall("equality/weld")), 3)
            tabletop = MuJoCoTabletop(scene, object_bodies=bodies)
            try:
                self.assertEqual(tabletop.get_obj_names(), names)
                for name in names:
                    self.assertLess(np.linalg.norm(tabletop.get_obj_pos(name) - positions[name]), 0.004)
            finally:
                tabletop.disconnect()

    def test_rejects_overlapping_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "invalid.xml"
            with self.assertRaisesRegex(ValueError, "15 cm"):
                write_scene({"blue block": (-0.1, 0), "red block": (-0.05, 0)}, scene)
            self.assertFalse(scene.exists())

    def test_four_block_ordinal_scene_compiles(self):
        from paper_suite import SEEN_UNSEEN, instantiate

        case = next(instantiate("seen_8", SEEN_UNSEEN, seed) for seed in range(100)
                    if instantiate("seen_8", SEEN_UNSEEN, seed).params["ordinal"] == "fourth")
        with tempfile.TemporaryDirectory() as temporary:
            scene = Path(temporary) / "four_blocks.xml"
            bodies = write_scene(sample_positions(case.objects, case.seed), scene)
            tabletop = MuJoCoTabletop(scene, object_bodies=bodies)
            try:
                self.assertEqual(sum(name.endswith(" block") for name in tabletop.get_obj_names()), 4)
                self.assertEqual(sum(tabletop.backend.model.eq(index).name.endswith("_grasp")
                                     for index in range(tabletop.backend.model.neq)), 4)
            finally:
                tabletop.disconnect()


if __name__ == "__main__":
    unittest.main()
