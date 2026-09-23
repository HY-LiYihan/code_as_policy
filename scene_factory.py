"""Build reproducible Piper tabletop scenes from the released simulation objects."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


TEMPLATE = Path(__file__).resolve().parent / "scenes/piper_tabletop.xml"
COLORS = {
    "blue": (78, 121, 167),
    "red": (255, 87, 89),
    "green": (89, 169, 79),
    "orange": (242, 142, 43),
    "yellow": (237, 201, 72),
    "purple": (176, 122, 161),
    "pink": (255, 157, 167),
    "cyan": (118, 183, 178),
    "brown": (156, 117, 95),
    "gray": (186, 176, 172),
}
FORWARD_OFFSET_METERS = 0.0
FORWARD_OFFSET = np.array([FORWARD_OFFSET_METERS, 0.0])
WORKSPACE_MIN = np.array([-0.19, -0.19]) + FORWARD_OFFSET
WORKSPACE_MAX = np.array([0.01, 0.19]) + FORWARD_OFFSET
PLACEMENT_SLOTS = np.array([
    [-0.18, -0.17], [-0.18, 0.0], [-0.18, 0.17],
    [0.0, -0.17], [0.0, 0.0], [0.0, 0.17],
]) + FORWARD_OFFSET


def sample_positions(objects: tuple[str, ...], seed: int) -> dict[str, tuple[float, float]]:
    if not 1 <= len(objects) <= len(PLACEMENT_SLOTS) or len(objects) != len(set(objects)):
        raise ValueError("A scene needs one to six distinct objects")
    generator = np.random.default_rng(seed)
    slot_indices = generator.permutation(len(PLACEMENT_SLOTS))[:len(objects)]
    positions = PLACEMENT_SLOTS[slot_indices] + generator.uniform(-0.008, 0.008, (len(objects), 2))
    return {name: tuple(float(value) for value in position)
            for name, position in zip(objects, positions)}


def write_scene(positions: dict[str, tuple[float, float]], destination: Path) -> dict[str, str]:
    if not positions or len(positions) > len(PLACEMENT_SLOTS):
        raise ValueError("A scene needs one to six objects")
    coordinates = np.asarray(list(positions.values()), dtype=float)
    if coordinates.shape != (len(positions), 2) or not np.isfinite(coordinates).all():
        raise ValueError("Every object needs finite XY coordinates")
    if np.any(coordinates < WORKSPACE_MIN) or np.any(coordinates > WORKSPACE_MAX):
        raise ValueError("Object position is outside the Piper task workspace")
    distances = np.linalg.norm(coordinates[:, None] - coordinates[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    if np.min(distances) < 0.15:
        raise ValueError("Objects must start at least 15 cm apart")

    tree = ET.parse(TEMPLATE)
    root = tree.getroot()
    world = root.find("worldbody")
    equality = root.find("equality")
    block = copy.deepcopy(world.find("body[@name='blue_block']"))
    bowl = copy.deepcopy(world.find("body[@name='yellow_bowl']"))
    for child in list(world):
        if child.tag == "body" and child.get("name", "").endswith(("_block", "_bowl")):
            world.remove(child)
    equality.clear()
    bodies = {}
    for name, xy in positions.items():
        parts = name.split()
        if len(parts) != 2 or parts[0] not in COLORS or parts[1] not in ("block", "bowl"):
            raise ValueError(f"Unknown tabletop object: {name}")
        body_name = name.replace(" ", "_")
        prototype = block if parts[1] == "block" else bowl
        body = copy.deepcopy(prototype)
        body.set("name", body_name)
        body.set("pos", f"{xy[0]:.9f} {xy[1]:.9f} {0.773 if parts[1] == 'block' else 0.75}")
        rgba = " ".join(str(component / 255) for component in COLORS[parts[0]]) + " 1"
        for geom in body.findall("geom"):
            geom.attrib.pop("name", None)
            geom.set("rgba", rgba)
        world.append(body)
        bodies[name] = body_name
        if parts[1] == "block":
            ET.SubElement(equality, "weld", {"name": f"{body_name}_grasp",
                                                "body1": "gripper_base", "body2": body_name,
                                                "active": "false"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return bodies
