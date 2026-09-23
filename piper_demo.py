"""Run a minimal Code as Policies tabletop task in the Piper MuJoCo simulator."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from robot_control import PiperRobot
from robot_control.api.types import Pose
from robot_control.backends.mujoco import MujocoBackend


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "scenes/piper_tabletop.xml"
NOTEBOOK = ROOT / "original" / "notebooks" / "Interactive_Demo.ipynb"
DOWN_QUATERNION = (0.0, 0.0, 1.0, 0.0)
START_JOINT_DEGREES = (0.0, 30.0, -45.0, 0.0, 60.0, 0.0)
START_JOINTS = np.deg2rad(START_JOINT_DEGREES)
OBJECT_BODIES = {
    "blue block": "blue_block",
    "green block": "green_block",
    "red block": "red_block",
    "yellow bowl": "yellow_bowl",
    "blue bowl": "blue_bowl",
    "green bowl": "green_bowl",
}


def upstream_prompt() -> str:
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.startswith("prompt_tabletop_ui = "):
            continue
        assignment = ast.parse(source).body[0]
        expression = assignment.value
        if (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Attribute)
                and expression.func.attr == "strip" and not expression.args
                and not expression.keywords):
            expression = expression.func.value
        return ast.literal_eval(expression).strip()
    raise ValueError("The upstream tabletop prompt is missing")


def parse_action(code: str, object_names: tuple[str, ...]) -> tuple[str, str]:
    code = code.strip()
    if code.startswith("```python") and code.endswith("```"):
        code = code[len("```python"):-3].strip()
    statements = ast.parse(code).body
    if len(statements) not in (1, 2):
        raise ValueError("Expected one pick-and-place call and, optionally, a say call")
    if len(statements) == 2:
        message = statements.pop(0)
        if not (isinstance(message, ast.Expr) and isinstance(message.value, ast.Call)
                and isinstance(message.value.func, ast.Name) and message.value.func.id == "say"
                and len(message.value.args) == 1 and isinstance(message.value.args[0], ast.Constant)
                and isinstance(message.value.args[0].value, str) and not message.value.keywords):
            raise ValueError("Only an optional say('message') call is supported")
    action = statements[0]
    if not (isinstance(action, ast.Expr) and isinstance(action.value, ast.Call)
            and isinstance(action.value.func, ast.Name)
            and action.value.func.id == "put_first_on_second"
            and len(action.value.args) == 2 and not action.value.keywords
            and all(isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    for arg in action.value.args)):
        raise ValueError("Expected put_first_on_second('object', 'target')")
    source, target = (arg.value for arg in action.value.args)
    if source not in object_names or target not in object_names or source == target:
        raise ValueError("The generated action must name two different scene objects")
    if source != "blue block":
        raise ValueError("Only the movable blue block can be picked up")
    return source, target


def generate_action(command: str, object_names: tuple[str, ...], model: str) -> str:
    from openai_client import create_openai_client

    client = create_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": upstream_prompt() + "\n"
             "Output only one Python call to put_first_on_second('object', 'target'). "
             "Use only objects in the user's list; do not write imports or other code."},
            {"role": "user", "content": f"objects = {list(object_names)!r}\n# {command}"},
        ],
    )
    code = response.choices[0].message.content
    if not code:
        raise ValueError("The model did not return an action")
    parse_action(code, object_names)
    return code


def pick_place_waypoints(source_xy, target_xy, stacking):
    approach_height = 0.97
    grasp_height = 0.92
    release_height = 0.965 if stacking else 0.93
    carry_height = 0.97 if stacking else 0.96
    return (
        (*source_xy, approach_height),
        (*source_xy, grasp_height),
        (*source_xy, carry_height),
        (*target_xy, carry_height),
        (*target_xy, release_height),
        (*target_xy, carry_height),
    )


class MuJoCoTabletop:
    def __init__(self, scene: Path, wrist_camera: bool = True,
                 object_bodies: dict[str, str] | None = None,
                 initial_joints=START_JOINTS):
        self.object_bodies = OBJECT_BODIES if object_bodies is None else dict(object_bodies)
        self.backend = MujocoBackend(scene=scene, wrist_camera=wrist_camera,
                                     )
        self.backend.connect()
        self.backend.data.qpos[self.backend._arm_qpos] = np.asarray(initial_joints, dtype=float)
        self.backend.data.ctrl[self.backend._arm_actuators] = np.asarray(initial_joints, dtype=float)
        import mujoco

        mujoco.mj_forward(self.backend.model, self.backend.data)
        self.robot = PiperRobot(self.backend)
        self.perception = None
        self._rgbd_camera = None

    def disconnect(self) -> None:
        if self._rgbd_camera is not None:
            self._rgbd_camera.disconnect()
            self._rgbd_camera = None
        self.robot.disconnect()

    def read_rgbd(self, width: int = 1280, height: int = 720):
        from robot_control.sensors.mujoco_rgbd import MujocoRGBDCamera

        if self._rgbd_camera is None:
            self._rgbd_camera = MujocoRGBDCamera(
                self.backend.model, self.backend.data, width=width, height=height)
            self._rgbd_camera.connect()
        return self._rgbd_camera.read()

    def wrist_camera(self, width: int = 1280, height: int = 720):
        self.read_rgbd(width, height)
        return self._rgbd_camera

    def attach_wrist_sam3_perception(self, texts=None, host=None, port=None,
                                     confidence: float = 0.25):
        from rgbd_perception import mujoco_world_from_camera
        from sam3_client import SAM3TextSegmenter

        prompts = list(texts or self.object_bodies)
        segmenter_kwargs = {"texts": prompts, "confidence": confidence}
        if host is not None:
            segmenter_kwargs["host"] = host
        if port is not None:
            segmenter_kwargs["port"] = port
        segmenter = SAM3TextSegmenter(**segmenter_kwargs)
        return self.attach_perception(self.wrist_camera(), segmenter,
                                      lambda frame: mujoco_world_from_camera(self.backend, frame))

    def attach_perception(self, camera, segmenter, world_from_camera):
        from rgbd_perception import RGBDPerception

        self.perception = RGBDPerception(camera, segmenter, world_from_camera)
        return self.perception

    def get_obj_names(self) -> tuple[str, ...]:
        if self.perception is not None:
            return self.perception.get_obj_names()
        return tuple(self.object_bodies)

    def _object_xyz(self, name: str) -> np.ndarray:
        body = self.backend.model.body(self.object_bodies[name])
        self.robot.state()
        return self.backend.data.xpos[body.id].copy()

    def get_obj_pos(self, name: str) -> np.ndarray:
        if self.perception is not None:
            return self.perception.get_obj_pos(name)
        return self._object_xyz(name)[:2]

    def get_bbox(self, name: str) -> tuple[float, float, float, float]:
        if self.perception is not None:
            return self.perception.get_bbox(name)
        import mujoco

        body_id = self.backend.model.body(self.object_bodies[name]).id
        self.robot.state()
        bounds = []
        for geom_id in range(self.backend.model.ngeom):
            if self.backend.model.geom_bodyid[geom_id] != body_id:
                continue
            geom = self.backend.model.geom(geom_id)
            if geom.type == mujoco.mjtGeom.mjGEOM_BOX:
                rotation = self.backend.data.geom_xmat[geom_id].reshape(3, 3)
                radius_xy = np.abs(rotation[:2]) @ geom.size
            else:
                radius_xy = np.repeat(geom.size[0], 2)
            center_xy = self.backend.data.geom_xpos[geom_id][:2]
            bounds.append((center_xy - radius_xy, center_xy + radius_xy))
        if not bounds:
            raise ValueError(f"No geometry for {name}")
        lower = np.min([lower for lower, _ in bounds], axis=0)
        upper = np.max([upper for _, upper in bounds], axis=0)
        return float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])

    def get_robot_pos(self) -> np.ndarray:
        return np.asarray(self.robot.state().pose.position)

    def move_to(self, xyz: tuple[float, float, float], quaternion: tuple[float, ...]) -> None:
        self.robot.move_p(Pose(xyz, quaternion))
        self.robot.step(1200)
        measured = self.get_robot_pos()
        if np.linalg.norm(measured - xyz) > 0.01:
            raise RuntimeError(f"Robot did not reach {xyz}: measured {measured}")

    def _gripper_contact(self, source: str) -> bool:
        finger_names = {"gripper_link1", "gripper_link2"}
        touching = set()
        for contact in self.backend.data.contact[:self.backend.data.ncon]:
            bodies = {
                self.backend.model.body(self.backend.model.geom_bodyid[int(geom_id)]).name
                for geom_id in (contact.geom1, contact.geom2)
            }
            if self.object_bodies[source] in bodies:
                touching.update(bodies & finger_names)
        return touching == finger_names

    def _set_grasp(self, source: str, active: bool) -> None:
        import mujoco

        weld_id = self.backend.model.eq(f"{self.object_bodies[source]}_grasp").id
        if active:
            if not self._gripper_contact(source):
                raise RuntimeError(f"Both gripper fingers must touch {source} before grasping")
            parent_id = self.backend.model.body("gripper_base").id
            object_id = self.backend.model.body(self.object_bodies[source]).id
            parent_rotation = self.backend.data.xmat[parent_id].reshape(3, 3)
            relative_position = parent_rotation.T @ (
                self.backend.data.xpos[object_id] - self.backend.data.xpos[parent_id]
            )
            conjugate = np.empty(4)
            relative_quaternion = np.empty(4)
            mujoco.mju_negQuat(conjugate, self.backend.data.xquat[parent_id])
            mujoco.mju_mulQuat(relative_quaternion, conjugate, self.backend.data.xquat[object_id])
            self.backend.model.eq_data[weld_id, 3:6] = relative_position
            self.backend.model.eq_data[weld_id, 6:10] = relative_quaternion
        self.backend.data.eq_active[weld_id] = active
        mujoco.mj_forward(self.backend.model, self.backend.data)

    def put_first_on_second(self, source: str, target) -> None:
        if source not in self.object_bodies or not source.endswith(" block"):
            raise ValueError("Source must be a block in the current scene")
        if isinstance(target, str):
            if target not in self.object_bodies or target == source:
                raise ValueError("Target must be another scene object")
            target_xy = self.get_obj_pos(target)
            stacking = target.endswith(" block")
        else:
            target_xy = np.asarray(target, dtype=float)
            if target_xy.shape != (2,) or not np.isfinite(target_xy).all():
                raise ValueError("Target position must be a finite XY pair")
            if abs(target_xy[0]) > 0.5 or abs(target_xy[1]) > 0.3:
                raise ValueError("Target position is outside the table")
            stacking = False
        source_xy = self.get_obj_pos(source)
        waypoints = pick_place_waypoints(source_xy, target_xy, stacking)

        self.robot.gripper(0.08)
        self.robot.wait_until_idle()
        self.move_to(waypoints[0], DOWN_QUATERNION)
        self.move_to(waypoints[1], DOWN_QUATERNION)
        self.robot.gripper(0.0)
        self.robot.step(500)
        self._set_grasp(source, True)
        try:
            self.move_to(waypoints[2], DOWN_QUATERNION)
            if self._object_xyz(source)[2] < 0.79:
                raise RuntimeError("The block did not lift off the table")
            self.move_to(waypoints[3], DOWN_QUATERNION)
            self.move_to(waypoints[4], DOWN_QUATERNION)
        finally:
            self._set_grasp(source, False)
            self.robot.gripper(0.08)
            self.robot.step(700)
        self.move_to(waypoints[5], DOWN_QUATERNION)
        self.robot.step(300)
        result = self._object_xyz(source)
        if np.linalg.norm(result[:2] - target_xy) > 0.03:
            raise RuntimeError("The block was not deposited at the target")
        if stacking and result[2] < self._object_xyz(target)[2] + 0.036:
            raise RuntimeError("The block did not stay on top of its target")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--command", default="put the blue block on the yellow bowl")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--offline-code", help="Use an already generated action without contacting an API")
    parser.add_argument("--execute", action="store_true", help="Execute the planned pick-and-place")
    args = parser.parse_args()

    if args.offline_code is None:
        code = generate_action(args.command, tuple(OBJECT_BODIES), args.model)
    else:
        code = args.offline_code
    source, target = parse_action(code, tuple(OBJECT_BODIES))
    print(f"Google-style action: {code.strip()}")
    if not args.execute:
        return

    tabletop = MuJoCoTabletop(args.scene)
    try:
        tabletop.put_first_on_second(source, target)
        print("Object position after action:", tabletop._object_xyz(source))
    finally:
        tabletop.disconnect()


if __name__ == "__main__":
    main()
