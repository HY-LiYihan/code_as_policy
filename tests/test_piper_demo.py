import unittest
import numpy as np

from piper_demo import (DOWN_QUATERNION, START_JOINT_DEGREES, START_JOINTS,
                        MuJoCoTabletop, SCENE, parse_action, pick_place_waypoints,
                        upstream_prompt)


class PromptTests(unittest.TestCase):
    def test_google_prompt_is_read_from_unchanged_notebook(self):
        self.assertIn("put_first_on_second('yellow block', 'yellow bowl')", upstream_prompt())

    def test_only_known_pick_place_action_is_accepted(self):
        objects = ("blue block", "yellow bowl")
        self.assertEqual(
            parse_action("say('Okay')\nput_first_on_second('blue block', 'yellow bowl')", objects),
            objects,
        )
        for code in (
            "import os\nput_first_on_second('blue block', 'yellow bowl')",
            "put_first_on_second('red cube', 'yellow bowl')",
            "__import__('os').system('echo nope')",
            "while True: pass",
        ):
            with self.subTest(code=code), self.assertRaises(ValueError):
                parse_action(code, objects)


class SimulationTests(unittest.TestCase):
    def test_default_initial_joints_match_requested_pose(self):
        np.testing.assert_allclose(START_JOINT_DEGREES, [0.0, 30.0, -45.0, 0.0, 60.0, 0.0])
        tabletop = MuJoCoTabletop(SCENE)
        try:
            np.testing.assert_allclose(tabletop.backend.data.qpos[tabletop.backend._arm_qpos], START_JOINTS)
        finally:
            tabletop.disconnect()

    def test_optional_perception_replaces_public_object_queries(self):
        class Observations:
            def get_obj_names(self):
                return ("blue block",)

            def get_obj_pos(self, name):
                return np.array([0.1, 0.2])

            def get_bbox(self, name):
                return (0.08, 0.18, 0.12, 0.22)

        tabletop = MuJoCoTabletop(SCENE)
        try:
            tabletop.perception = Observations()
            self.assertEqual(tabletop.get_obj_names(), ("blue block",))
            np.testing.assert_allclose(tabletop.get_obj_pos("blue block"), [0.1, 0.2])
            self.assertEqual(tabletop.get_bbox("blue block"), (0.08, 0.18, 0.12, 0.22))
        finally:
            tabletop.disconnect()

    def test_all_pick_place_waypoints_share_downward_tool_orientation(self):
        import mujoco

        downward = np.empty(3)
        mujoco.mju_rotVecQuat(downward, np.array([0.0, 0.0, 1.0]), np.asarray(DOWN_QUATERNION))
        np.testing.assert_allclose(downward, [0.0, 0.0, -1.0])
        waypoints = pick_place_waypoints((0.0, -0.19), (0.0, 0.19), stacking=True)
        self.assertEqual(len(waypoints), 6)
        self.assertEqual([waypoint[2] for waypoint in waypoints],
                         [0.97, 0.92, 0.97, 0.97, 0.965, 0.97])

    def test_rotated_block_bbox_uses_world_aabb(self):
        import mujoco

        tabletop = MuJoCoTabletop(SCENE)
        try:
            body = tabletop.backend.model.body("blue_block")
            joint_id = tabletop.backend.model.body_jntadr[body.id]
            quaternion_start = tabletop.backend.model.jnt_qposadr[joint_id] + 3
            tabletop.backend.data.qpos[quaternion_start:quaternion_start + 4] = [
                np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8),
            ]
            mujoco.mj_forward(tabletop.backend.model, tabletop.backend.data)
            min_x, min_y, max_x, max_y = tabletop.get_bbox("blue block")
            expected_width = 2 * 0.022 * np.sqrt(2)
            self.assertAlmostEqual(max_x - min_x, expected_width, places=3)
            self.assertAlmostEqual(max_y - min_y, expected_width, places=3)
        finally:
            tabletop.disconnect()

    def test_scene_grasp_and_release(self):
        tabletop = MuJoCoTabletop(SCENE)
        try:
            self.assertEqual(
                tabletop.get_obj_names(),
                ("blue block", "green block", "red block", "yellow bowl", "blue bowl", "green bowl"),
            )
            initial = tabletop.get_obj_pos("blue block")
            target = tabletop.get_obj_pos("yellow bowl")
            self.assertGreater(np.linalg.norm(initial - target), 0.1)
            self.assertEqual(len(tabletop.get_bbox("yellow bowl")), 4)
            tabletop.put_first_on_second("blue block", "yellow bowl")
            self.assertLess(np.linalg.norm(tabletop.get_obj_pos("blue block") - target), 0.03)
            self.assertFalse(tabletop.backend.data.eq_active[tabletop.backend.model.eq("blue_block_grasp").id])
            self.assertGreater(tabletop._object_xyz("blue block")[2], 0.76)
        finally:
            tabletop.disconnect()

    def test_coordinate_target(self):
        tabletop = MuJoCoTabletop(SCENE)
        try:
            target = np.array([0.0, 0.2])
            tabletop.put_first_on_second("green block", target)
            self.assertLess(np.linalg.norm(tabletop.get_obj_pos("green block") - target), 0.03)
        finally:
            tabletop.disconnect()


if __name__ == "__main__":
    unittest.main()
