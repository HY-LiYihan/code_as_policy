import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from franka_real import (FR3_WORKSPACE_MAX, FR3_WORKSPACE_MIN,
                         FrankaRealLMPWrapper, FrankaRealTabletop,
                         load_observations)
from robot_control.api.types import Pose


class FrankaRealAdapterTests(unittest.TestCase):
    def setUp(self):
        self.observations = {
            "blue block": {"xyz": np.array([0.42, -0.12, 0.02]),
                           "bbox_xy": (0.4, -0.14, 0.44, -0.10),
                           "color_rgb": (0.1, 0.25, 0.9, 1.0)},
            "yellow bowl": {"xyz": np.array([0.50, 0.12, 0.0]),
                             "bbox_xy": (0.44, 0.06, 0.56, 0.18),
                             "color_rgb": (0.95, 0.8, 0.1, 1.0)},
        }

    def test_observations_load_in_base_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.json"
            path.write_text(json.dumps({"objects": {
                "blue block": {"xyz": [0.42, -0.12, 0.02]}
            }}))
            loaded = load_observations(path)
        np.testing.assert_allclose(loaded["blue block"]["xyz"], [0.42, -0.12, 0.02])

    def test_dry_run_never_connects_or_moves_hardware(self):
        tabletop = FrankaRealTabletop(self.observations, execute=False, table_z=0.0)
        tabletop.put_first_on_second("blue block", "yellow bowl")
        self.assertIsNone(tabletop.robot)
        self.assertEqual(len(tabletop.actions), 1)
        self.assertFalse(tabletop.actions[0]["executed"])
        self.assertAlmostEqual(tabletop.get_obj_pos("blue block")[0], 0.50)

    def test_execute_path_routes_pick_and_place_to_fr3_api(self):
        class FakeRobot:
            def __init__(self):
                self.gripper_calls = []
                self.pose_calls = []
                self.disconnected = False

            def gripper(self, width):
                self.gripper_calls.append(width)

            def move_p(self, pose):
                self.pose_calls.append(pose)

            def disconnect(self):
                self.disconnected = True

        fake_robot = FakeRobot()
        with patch("franka_real.Robot.connect", return_value=fake_robot) as connect:
            tabletop = FrankaRealTabletop(
                self.observations,
                execute=True,
                robot_ip="192.168.1.6",
                motion_duration_s=4.0,
                orientation=(1.0, 0.0, 0.0, 0.0),
            )
            tabletop.put_first_on_second("blue block", "yellow bowl")
            tabletop.disconnect()

        connect.assert_called_once_with(
            "real",
            robot="franka_fr3",
            config={
                "gripper_speed_m_s": 0.05,
                "robot_ip": "192.168.1.6",
                "motion_duration_s": 4.0,
            },
        )
        self.assertEqual(fake_robot.gripper_calls, [0.08, 0.0])
        self.assertEqual(len(fake_robot.pose_calls), 6)
        self.assertTrue(all(isinstance(pose, Pose) for pose in fake_robot.pose_calls))
        self.assertTrue(fake_robot.disconnected)

    def test_wrapper_uses_fr3_workspace_and_object_colors(self):
        tabletop = FrankaRealTabletop(self.observations, execute=False)
        wrapper = FrankaRealLMPWrapper(tabletop)
        np.testing.assert_allclose(wrapper.denormalize_xy([0, 0]), FR3_WORKSPACE_MIN)
        np.testing.assert_allclose(wrapper.denormalize_xy([1, 1]), FR3_WORKSPACE_MAX)
        self.assertEqual(wrapper.get_color("blue block"), (0.1, 0.25, 0.9, 1.0))

    def test_invalid_observation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "objects.json"
            path.write_text(json.dumps({"objects": {
                "blue block": {"xyz": [0.42, 0.12]}
            }}))
            with self.assertRaises(ValueError):
                load_observations(path)


if __name__ == "__main__":
    unittest.main()
