"""Camera-independent RGB-D object localization for simulation and hardware."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

import numpy as np

from robot_control.sensors.frame import RGBDFrame


class RGBDCamera(Protocol):
    def read(self) -> RGBDFrame: ...


class InstanceSegmenter(Protocol):
    def detect(self, rgb: np.ndarray) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class ObjectObservation:
    name: str
    xyz: tuple[float, float, float]
    bbox_xy: tuple[float, float, float, float]
    bbox_pixels: tuple[int, int, int, int]
    color_rgb: tuple[float, float, float, float]
    timestamp: float


def mujoco_world_from_camera(backend, frame: RGBDFrame) -> np.ndarray:
    base_from_camera = arm_origin_from_camera(frame)
    base = backend.model.body("base_link").id
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = backend.data.xmat[base].reshape(3, 3)
    world_from_base[:3, 3] = backend.data.xpos[base]
    return world_from_base @ base_from_camera


def arm_origin_from_camera(frame: RGBDFrame) -> np.ndarray:
    """Return camera-coordinate points in the arm-origin coordinate system."""
    extrinsics = frame.extrinsics
    if extrinsics is None or extrinsics.reference_frame not in ("base_link", "arm_origin"):
        raise ValueError("RGB-D frame must provide camera pose relative to the arm origin")
    if extrinsics.camera_frame != frame.frame_id:
        raise ValueError("RGB-D extrinsics camera frame does not match frame_id")
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(extrinsics.rotation).reshape(3, 3)
    transform[:3, 3] = extrinsics.translation
    return transform


class RGBDPerception:
    def __init__(self, camera: RGBDCamera, segmenter: InstanceSegmenter,
                 world_from_camera: Callable[[RGBDFrame], np.ndarray],
                 min_depth: float = 0.05, max_depth: float = 3.0,
                 depth_mad_scale: float = 3.5,
                 minimum_depth_tolerance: float = 0.005):
        if not 0 < min_depth < max_depth:
            raise ValueError("Depth limits must be positive and increasing")
        if depth_mad_scale <= 0 or minimum_depth_tolerance <= 0:
            raise ValueError("Depth outlier parameters must be positive")
        self.camera = camera
        self.segmenter = segmenter
        self.world_from_camera = world_from_camera
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_mad_scale = depth_mad_scale
        self.minimum_depth_tolerance = minimum_depth_tolerance
        self.observations: dict[str, ObjectObservation] | None = None

    def refresh(self) -> dict[str, ObjectObservation]:
        frame = self.camera.read()
        transform = np.asarray(self.world_from_camera(frame), dtype=float)
        if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
            raise ValueError("A calibrated world-from-camera transform is required")
        intrinsics = frame.intrinsics
        if (frame.color.shape[:2] != (intrinsics.height, intrinsics.width)
                or min(intrinsics.fx, intrinsics.fy) <= 0 or frame.depth_scale <= 0):
            raise ValueError("Camera calibration does not match the RGB-D frame")
        depth = np.asarray(frame.depth, dtype=float) * frame.depth_scale
        observations = {}
        for name, mask in self.segmenter.detect(frame.color).items():
            mask = np.asarray(mask)
            if mask.shape != depth.shape or mask.dtype != np.bool_:
                raise ValueError(f"Object mask must be an HxW boolean array: {name}")
            valid = mask & np.isfinite(depth) & (depth >= self.min_depth) & (depth <= self.max_depth)
            rows, columns = np.nonzero(valid)
            if len(rows) == 0:
                continue
            distances = depth[rows, columns]
            median_depth = float(np.median(distances))
            absolute_deviation = np.abs(distances - median_depth)
            mad = float(np.median(absolute_deviation))
            tolerance = max(self.minimum_depth_tolerance,
                            self.depth_mad_scale * 1.4826 * mad)
            inliers = absolute_deviation <= tolerance
            rows = rows[inliers]
            columns = columns[inliers]
            distances = distances[inliers]
            if len(rows) == 0:
                continue
            xyz_camera = np.stack(((columns - intrinsics.cx) * distances / intrinsics.fx,
                                   (rows - intrinsics.cy) * distances / intrinsics.fy,
                                   distances), axis=1)
            xyz_world = xyz_camera @ transform[:3, :3].T + transform[:3, 3]
            position = np.median(xyz_world, axis=0)
            minimum = xyz_world[:, :2].min(axis=0)
            maximum = xyz_world[:, :2].max(axis=0)
            color = np.median(frame.color[rows, columns], axis=0) / 255
            observations[name] = ObjectObservation(
                name, tuple(float(value) for value in position),
                (float(minimum[0]), float(minimum[1]), float(maximum[0]), float(maximum[1])),
                (int(columns.min()), int(rows.min()), int(columns.max()) + 1, int(rows.max()) + 1),
                tuple(float(value) for value in color) + (1.0,), frame.timestamp,
            )
        self.observations = observations
        return observations

    def _observations(self) -> dict[str, ObjectObservation]:
        if self.observations is None:
            return self.refresh()
        return self.observations

    def get_obj_names(self) -> tuple[str, ...]:
        return tuple(self._observations())

    def is_obj_visible(self, name: str) -> bool:
        return name in self._observations()

    def get_obj_pos(self, name: str) -> np.ndarray:
        return np.asarray(self._observations()[name].xyz[:2])

    def get_bbox(self, name: str) -> tuple[float, float, float, float]:
        return self._observations()[name].bbox_xy

    def get_color(self, name: str) -> tuple[float, float, float, float]:
        return self._observations()[name].color_rgb
