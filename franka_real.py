"""Safety-gated Franka FR3 tabletop adapter for the original Code as Policies LMP."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from robot_control import Robot
from robot_control.api.types import Pose

from piper_lmp import MuJoCoLMPWrapper


FR3_WORKSPACE_MIN = np.array([0.20, -0.45], dtype=float)
FR3_WORKSPACE_MAX = np.array([0.75, 0.45], dtype=float)
DEFAULT_BLOCK_HEIGHT = 0.044
DEFAULT_GRIPPER_OPENING = 0.08


def load_observations(path: Path) -> dict[str, dict]:
    """Load object observations expressed in the Franka base frame."""
    payload = json.loads(path.read_text())
    objects = payload.get("objects", payload)
    if not isinstance(objects, dict) or not objects:
        raise ValueError("observation JSON must contain a non-empty object mapping")
    normalized = {}
    for name, value in objects.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(value, dict):
            raise ValueError("each observation must map a name to an object record")
        xyz = np.asarray(value.get("xyz"), dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError(f"{name}: xyz must contain three finite base-frame metres")
        record = dict(value)
        record["xyz"] = xyz
        record.setdefault("bbox_xy", [float(xyz[0]), float(xyz[1]), float(xyz[0]), float(xyz[1])])
        record.setdefault("color_rgb", [0.5, 0.5, 0.5, 1.0])
        normalized[name] = record
    return normalized


class FrankaRealTabletop:
    """Object map plus real FR3 motion primitives.

    Object coordinates must already be in the FR3 base frame. The class does
    not infer camera extrinsics and never connects to hardware unless execute
    is true.
    """

    def __init__(self, observations: dict[str, dict], *, execute: bool = False,
                 robot_ip: str | None = None, motion_duration_s: float | None = None,
                 gripper_speed_m_s: float = 0.05, table_z: float = 0.0,
                 grasp_clearance_m: float = 0.02, block_height_m: float = DEFAULT_BLOCK_HEIGHT,
                 orientation: tuple[float, float, float, float] | None = None):
        self.observations = observations
        self.execute = bool(execute)
        self.table_z = float(table_z)
        self.grasp_clearance_m = float(grasp_clearance_m)
        self.block_height_m = float(block_height_m)
        self.robot = None
        self.actions = []
        self.orientation = orientation
        if not all(math.isfinite(value) for value in (
                self.table_z, self.grasp_clearance_m, self.block_height_m)):
            raise ValueError("table and clearance parameters must be finite")
        if self.execute:
            config = {"gripper_speed_m_s": gripper_speed_m_s}
            if robot_ip is not None:
                config["robot_ip"] = robot_ip
            if motion_duration_s is not None:
                config["motion_duration_s"] = motion_duration_s
            self.robot = Robot.connect("real", robot="franka_fr3", config=config)
            if self.orientation is None:
                state = self.robot.state()
                if state.pose is None:
                    self.robot.disconnect()
                    self.robot = None
                    raise RuntimeError("Franka state did not include an end-effector pose")
                self.orientation = state.pose.quaternion
        self.orientation = tuple(self.orientation or (1.0, 0.0, 0.0, 0.0))
        if len(self.orientation) != 4 or not np.isfinite(self.orientation).all():
            raise ValueError("orientation must contain four finite values")

    def disconnect(self) -> None:
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    def get_obj_names(self):
        return tuple(self.observations)

    def get_obj_pos(self, name):
        return np.asarray(self.observations[name]["xyz"], dtype=float)[:2]

    def _object_xyz(self, name):
        return np.asarray(self.observations[name]["xyz"], dtype=float).copy()

    def get_bbox(self, name):
        return tuple(float(value) for value in self.observations[name]["bbox_xy"])

    def get_color(self, name):
        return tuple(float(value) for value in self.observations[name]["color_rgb"])

    def _pose(self, xyz):
        return Pose(tuple(float(value) for value in xyz), self.orientation)

    def _waypoints(self, source, target):
        source_xyz = self._object_xyz(source)
        target_xyz = self._object_xyz(target) if isinstance(target, str) else np.array(
            [float(target[0]), float(target[1]), self.table_z], dtype=float)
        source_grasp = max(float(source_xyz[2]) + self.grasp_clearance_m, self.table_z + 0.01)
        target_surface = float(target_xyz[2])
        release_z = target_surface + (self.block_height_m if isinstance(target, str)
                                      and target.endswith(" block") else self.grasp_clearance_m)
        carry_z = max(source_grasp + 0.10, release_z + 0.10)
        return [
            np.array([source_xyz[0], source_xyz[1], carry_z]),
            np.array([source_xyz[0], source_xyz[1], source_grasp]),
            np.array([source_xyz[0], source_xyz[1], carry_z]),
            np.array([target_xyz[0], target_xyz[1], carry_z]),
            np.array([target_xyz[0], target_xyz[1], release_z]),
            np.array([target_xyz[0], target_xyz[1], carry_z]),
        ]

    def put_first_on_second(self, source, target):
        if source not in self.observations or not source.endswith(" block"):
            raise ValueError("source must be a visible block")
        if isinstance(target, str) and target not in self.observations:
            raise ValueError("target must be a visible object")
        waypoints = self._waypoints(source, target)
        if not self.execute:
            self.actions.append({"source": source, "target": target,
                                 "waypoints": [point.tolist() for point in waypoints],
                                 "executed": False})
            self.observations[source]["xyz"] = waypoints[4].copy()
            return
        if self.robot is None:
            raise RuntimeError("real Franka connection is not available")
        self.robot.gripper(DEFAULT_GRIPPER_OPENING)
        for point in waypoints[:2]:
            self.robot.move_p(self._pose(point))
        self.robot.gripper(0.0)
        for point in waypoints[2:]:
            self.robot.move_p(self._pose(point))
        self.actions.append({"source": source, "target": target,
                             "waypoints": [point.tolist() for point in waypoints],
                             "executed": True})
        self.observations[source]["xyz"] = waypoints[4].copy()


class FrankaRealLMPWrapper(MuJoCoLMPWrapper):
    """Reuse the official tabletop helper surface with FR3 coordinates."""

    def __init__(self, tabletop: FrankaRealTabletop):
        super().__init__(tabletop, min_xy=FR3_WORKSPACE_MIN, max_xy=FR3_WORKSPACE_MAX)

    def get_color(self, name):
        return self.tabletop.get_color(name)

    def put_first_on_second(self, source, target):
        self.tabletop.put_first_on_second(source, target)
        self.actions.append((source, target, True))


def plan_from_observations(observations: dict[str, dict], command: str, *, client,
                           model: str, execute: bool = False, **kwargs) -> tuple[FrankaRealTabletop, object]:
    from piper_lmp import setup_original_lmp

    tabletop = FrankaRealTabletop(observations, execute=execute, **kwargs)
    policy, wrapper = setup_original_lmp(
        tabletop, client, model, protocol="paper-sim",
        wrapper_cls=FrankaRealLMPWrapper,
    )
    policy(command.rstrip("."), f"objects = {list(observations)!r}")
    return tabletop, wrapper
