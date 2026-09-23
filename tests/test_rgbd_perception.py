import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_control.sensors.frame import CameraExtrinsics, CameraIntrinsics, RGBDFrame

from rgbd_perception import RGBDPerception, mujoco_world_from_camera


class FixedCamera:
    def __init__(self, frame):
        self.frame = frame
        self.reads = 0

    def read(self):
        self.reads += 1
        return self.frame


class FixedSegmenter:
    def __init__(self, masks):
        self.masks = masks

    def detect(self, rgb):
        return self.masks


class RGBDPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.color = np.zeros((3, 4, 3), dtype=np.uint8)
        self.color[1, 2] = (200, 100, 50)
        self.depth = np.zeros((3, 4), dtype=np.uint16)
        self.depth[1, 2] = 1000
        self.mask = np.zeros((3, 4), dtype=bool)
        self.mask[1, 2] = True
        self.frame = RGBDFrame(self.color, self.depth, 123.0, "color_optical",
                               CameraIntrinsics(4, 3, 10, 10, 1, 1), 0.001)
        self.world_from_camera = np.eye(4)
        self.world_from_camera[:3, 3] = [0.3, -0.2, 0.75]

    def test_same_rgbd_pipeline_deprojects_real_depth_units(self):
        camera = FixedCamera(self.frame)
        perception = RGBDPerception(camera, FixedSegmenter({"orange block": self.mask}),
                                    lambda frame: self.world_from_camera)
        observation = perception.refresh()["orange block"]
        np.testing.assert_allclose(observation.xyz, [0.4, -0.2, 1.75])
        np.testing.assert_allclose(perception.get_obj_pos("orange block"), [0.4, -0.2])
        self.assertEqual(perception.get_bbox("orange block"), (0.4, -0.2, 0.4, -0.2))
        self.assertEqual(observation.bbox_pixels, (2, 1, 3, 2))
        np.testing.assert_allclose(perception.get_color("orange block"),
                                   [200 / 255, 100 / 255, 50 / 255, 1])
        self.assertEqual(perception.get_obj_names(), ("orange block",))
        self.assertFalse(perception.is_obj_visible("red block"))
        self.assertEqual(camera.reads, 1)

    def test_invalid_depth_is_not_mistaken_for_visible_object(self):
        self.depth[1, 2] = 0
        perception = RGBDPerception(FixedCamera(self.frame),
                                    FixedSegmenter({"orange block": self.mask}),
                                    lambda frame: self.world_from_camera)
        self.assertEqual(perception.refresh(), {})
        with self.assertRaises(KeyError):
            perception.get_obj_pos("orange block")

    def test_finite_depth_outliers_are_removed_from_all_object_statistics(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[1, 0:4] = True
        self.depth[1, 0:4] = [1000, 1000, 1000, 2500]
        self.color[1, 0:4] = (200, 100, 50)
        perception = RGBDPerception(FixedCamera(self.frame),
                                    FixedSegmenter({"orange block": mask}),
                                    lambda frame: self.world_from_camera)
        observation = perception.refresh()["orange block"]
        np.testing.assert_allclose(observation.xyz, [0.3, -0.2, 1.75])
        self.assertEqual(observation.bbox_pixels, (0, 1, 3, 2))
        np.testing.assert_allclose(observation.bbox_xy, (0.2, -0.2, 0.4, -0.2))

    def test_invalid_depth_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            RGBDPerception(FixedCamera(self.frame), FixedSegmenter({}),
                           lambda frame: self.world_from_camera,
                           depth_mad_scale=0)

    def test_mask_and_calibration_must_match_camera_frame(self):
        perception = RGBDPerception(FixedCamera(self.frame),
                                    FixedSegmenter({"orange block": self.mask[:2]}),
                                    lambda frame: self.world_from_camera)
        with self.assertRaisesRegex(ValueError, "boolean array"):
            perception.refresh()
        perception = RGBDPerception(FixedCamera(self.frame), FixedSegmenter({}),
                                    lambda frame: np.eye(3))
        with self.assertRaisesRegex(ValueError, "calibrated"):
            perception.refresh()

    def test_mujoco_pose_provider_converts_base_frame_to_world(self):
        extrinsics = CameraExtrinsics(tuple(np.eye(3).flat), (0.1, 0.2, 0.3),
                                      "base_link", "color_optical")
        self.frame.extrinsics = extrinsics
        backend = SimpleNamespace(
            model=SimpleNamespace(body=lambda name: SimpleNamespace(id=0)),
            data=SimpleNamespace(xmat=np.array([np.eye(3).flatten()]),
                                 xpos=np.array([[-0.3, 0.0, 0.75]])),
        )
        transform = mujoco_world_from_camera(backend, self.frame)
        np.testing.assert_allclose(transform[:3, 3], [-0.2, 0.2, 1.05])
        self.frame.extrinsics = None
        with self.assertRaisesRegex(ValueError, "camera pose"):
            mujoco_world_from_camera(backend, self.frame)

    def test_real_mujoco_wrist_rgbd_reprojects_visible_block(self):
        mjpython = Path(sys.executable).with_name("mjpython")
        if not mjpython.exists():
            self.skipTest("mjpython and offscreen rendering are required")
        with tempfile.TemporaryDirectory() as temporary:
            script = "\n".join([
                "import numpy as np",
                "from pathlib import Path",
                "from paper_suite import generate_cases",
                "from scene_factory import sample_positions, write_scene",
                "from piper_demo import MuJoCoTabletop",
                "from robot_control.sensors.mujoco_rgbd import MujocoRGBDCamera",
                "from rgbd_perception import RGBDPerception, mujoco_world_from_camera",
                "case = generate_cases(20260923)[0]",
                f"scene = Path({str(Path(temporary) / 'scene.xml')!r})",
                "bodies = write_scene(sample_positions(case.objects, (case.seed + 1) % 2**64), scene)",
                "tabletop = MuJoCoTabletop(scene, wrist_camera=True, object_bodies=bodies)",
                "camera = MujocoRGBDCamera(tabletop.backend.model, tabletop.backend.data, width=640, height=480)",
                "class Segmenter:",
                "    def detect(self, rgb):",
                "        red = rgb[:, :, 0].astype(float)",
                "        columns = np.arange(rgb.shape[1])[None, :]",
                "        return {'red block': (red > 120) & (red > rgb[:, :, 1] * 1.5) & (red > rgb[:, :, 2] * 1.3) & (columns > 440)}",
                "try:",
                "    camera.connect()",
                "    perception = RGBDPerception(camera, Segmenter(), lambda frame: mujoco_world_from_camera(tabletop.backend, frame))",
                "    detected = perception.refresh()['red block']",
                "    actual = tabletop._object_xyz('red block')",
                "    assert np.linalg.norm(np.array(detected.xyz) - actual) < 0.04, (detected, actual)",
                "finally:",
                "    camera.disconnect()",
                "    tabletop.disconnect()",
            ])
            subprocess.run([str(mjpython), "-c", script], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
