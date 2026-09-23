"""Probe the fixed downward-tool Piper workspace with the frozen backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from robot_control.api.types import Pose

from piper_demo import DOWN_QUATERNION, MuJoCoTabletop, SCENE
from repro_metadata import backend_metadata
from scene_factory import WORKSPACE_MAX, WORKSPACE_MIN


HEIGHTS = (0.92, 0.93, 0.965, 0.97)


def probe_workspace(x_min: float, x_max: float, y_min: float, y_max: float,
                    step: float) -> dict:
    if step <= 0 or x_min >= x_max or y_min >= y_max:
        raise ValueError("Workspace bounds and step must define a positive grid")
    tabletop = MuJoCoTabletop(SCENE, wrist_camera=False)
    points = []
    try:
        xs = np.arange(x_min, x_max + step / 2, step)
        ys = np.arange(y_min, y_max + step / 2, step)
        for x in xs:
            for y in ys:
                errors = []
                for height in HEIGHTS:
                    pose = tabletop.backend._transform_ik_pose(
                        Pose((float(x), float(y), height), DOWN_QUATERNION), inverse=True)
                    solution = tabletop.backend.ik.solve(
                        pose, seed=tabletop.backend.data.qpos[tabletop.backend._arm_qpos].copy())
                    if not solution.success:
                        errors.append({"height": height, "message": solution.message})
                points.append({"xy": [float(x), float(y)], "reachable": not errors,
                               "failures": errors})
    finally:
        tabletop.disconnect()
    reachable = [point["xy"] for point in points if point["reachable"]]
    return {
        "heights_m": list(HEIGHTS),
        "step_m": step,
        "bounds": {"min_xy": [x_min, y_min], "max_xy": [x_max, y_max]},
        "grid_points": len(points),
        "reachable_points": len(reachable),
        "reachable_min_xy": np.min(reachable, axis=0).tolist() if reachable else None,
        "reachable_max_xy": np.max(reachable, axis=0).tolist() if reachable else None,
        "points": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = probe_workspace(float(WORKSPACE_MIN[0]), float(WORKSPACE_MAX[0]),
                             float(WORKSPACE_MIN[1]), float(WORKSPACE_MAX[1]), args.step)
    result["backend"] = backend_metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"Calibration written to {args.output}")


if __name__ == "__main__":
    main()
