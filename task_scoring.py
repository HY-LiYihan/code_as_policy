"""Explicit, reconstructed physical success rules for the Piper tabletop suite."""

from __future__ import annotations

import numpy as np

from scene_factory import WORKSPACE_MAX, WORKSPACE_MIN


OFFSET_METERS = {"a little": 0.06, "a lot": 0.12}
POSITION_TOLERANCE = 0.045
CORNER_MARGIN = 0.025


def xy(positions, name):
    return np.asarray(positions[name], dtype=float)[:2]


def corner_positions():
    lower = WORKSPACE_MIN + CORNER_MARGIN
    upper = WORKSPACE_MAX - CORNER_MARGIN
    return np.array([
        [lower[0], lower[1]], [lower[0], upper[1]],
        [upper[0], lower[1]], [upper[0], upper[1]],
    ])


def side_contains(position, side):
    x_value, y_value = np.asarray(position)[:2]
    allowed = {
        "left": x_value <= WORKSPACE_MIN[0] + 0.065,
        "top": y_value >= WORKSPACE_MAX[1] - 0.065,
        "right": x_value >= WORKSPACE_MAX[0] - 0.065,
        "bottom": y_value <= WORKSPACE_MIN[1] + 0.065,
    }
    return all(allowed[part] for part in side.split("-"))


def chosen_block(case, initial):
    blocks = case.expected["blocks"]
    parameters = case.parameters
    if case.category == "seen_6":
        bowl_xy = xy(initial, parameters["bowl"])
        direction = parameters["direction"]
        axis = 1 if direction in ("top", "bottom") else 0
        sign = 1 if direction in ("top", "right") else -1
        candidates = [name for name in blocks
                      if sign * (xy(initial, name)[axis] - bowl_xy[axis]) > 0.02]
        return candidates[0] if len(candidates) == 1 else None
    if case.category == "seen_7":
        bowl_xy = xy(initial, parameters["bowl"])
        distances = [np.linalg.norm(xy(initial, name) - bowl_xy) for name in blocks]
        index = np.argmin(distances) if parameters["distance"] == "closest" else np.argmax(distances)
        return blocks[index]
    if case.category == "seen_8":
        direction = parameters["direction"]
        axis = 1 if direction in ("top", "bottom") else 0
        descending = direction in ("top", "right")
        ordered = sorted(blocks, key=lambda name: xy(initial, name)[axis], reverse=descending)
        index = ("first", "second", "third", "fourth").index(parameters["ordinal"])
        return ordered[index]
    raise ValueError(f"No initial-state block referent for {case.category}")


def scene_precondition(case, positions):
    if case.category == "seen_6":
        return chosen_block(case, positions) is not None
    if case.category == "unseen_4":
        target = offset_target(case, positions)
        return bool(np.all(target >= WORKSPACE_MIN)
                    and np.all(target <= WORKSPACE_MAX))
    return True


def offset_target(case, positions):
    direction = case.parameters["direction"]
    offset = OFFSET_METERS[case.parameters["magnitude"]]
    delta = {"left": (-offset, 0), "right": (offset, 0),
             "top": (0, offset), "bottom": (0, -offset)}[direction]
    return xy(positions, case.parameters["bowl"]) + delta


def near(first, second, tolerance=POSITION_TOLERANCE):
    return bool(np.linalg.norm(np.asarray(first)[:2] - np.asarray(second)[:2]) <= tolerance)


def stacked(blocks, final):
    ordered = sorted(blocks, key=lambda name: final[name][2])
    return all(near(final[lower], final[upper], tolerance=0.04)
               and final[upper][2] > final[lower][2] + 0.035
               for lower, upper in zip(ordered, ordered[1:]))


def score_case(case, initial, final):
    category = case.category
    blocks = case.expected["blocks"]
    params = case.parameters
    if category == "demo_corner":
        return near(final["green block"], corner_positions()[3], tolerance=0.055)
    if category == "demo_leftmost":
        leftmost = min(blocks, key=lambda name: xy(initial, name)[0])
        return near(final[leftmost], final["green bowl"])
    if category == "demo_farthest":
        bowls = case.expected["bowls"]
        farthest = max(bowls, key=lambda name: np.linalg.norm(xy(initial, name) - xy(initial, "red block")))
        return near(final["red block"], final[farthest])
    if category == "demo_bottom":
        return side_contains(final["pink block"], "bottom")
    if category == "demo_stack":
        return stacked(blocks, final) and side_contains(final["gray block"], "right") and all(
            final["gray block"][2] <= final[name][2] for name in blocks
        )
    if category == "seen_1":
        source, target = params["source"], params["target"]
        result = near(final[source], final[target], tolerance=0.035)
        if target.endswith(" block"):
            result = result and final[source][2] > final[target][2] + 0.035
        return bool(result)
    if category == "seen_2":
        return stacked(blocks, final)
    if category == "seen_3":
        return all(side_contains(final[name], params["side"]) for name in blocks)
    if category == "seen_4":
        return all(near(final[name], final[params["bowl"]]) for name in blocks)
    if category == "seen_5":
        return all(near(final[name], final[f"{name.split()[0]} bowl"]) for name in blocks)
    if category in ("seen_6", "seen_7", "seen_8"):
        source = chosen_block(case, initial)
        return source is not None and side_contains(final[source], params["side"])
    if category == "unseen_1":
        corners = corner_positions()
        nearest = [int(np.argmin(np.linalg.norm(corners - xy(final, name), axis=1))) for name in blocks]
        return len(set(nearest)) == len(blocks) and all(
            near(final[name], corners[index], tolerance=0.055) for name, index in zip(blocks, nearest)
        )
    if category == "unseen_2":
        selected = []
        for name in blocks:
            matching = [bowl for bowl in case.expected["bowls"]
                        if bowl.split()[0] != name.split()[0] and near(final[name], final[bowl])]
            if len(matching) != 1:
                return False
            selected.extend(matching)
        return len(selected) == len(set(selected))
    if category == "unseen_3":
        base = min(blocks, key=lambda name: final[name][2])
        return stacked(blocks, final) and side_contains(final[base], params["side"])
    if category == "unseen_4":
        return near(final[params["block"]], offset_target(case, initial))
    if category == "unseen_5":
        corners = corner_positions()
        distances = np.linalg.norm(corners - xy(initial, params["bowl"]), axis=1)
        index = np.argmin(distances) if params["distance"] == "closest" else np.argmax(distances)
        return near(final[params["block"]], corners[index], tolerance=0.055)
    if category == "unseen_6":
        points = np.array([xy(final, name) for name in blocks])
        direction = params["line"]
        spread = np.ptp(points, axis=0)
        if direction == "horizontal":
            return bool(spread[0] >= 0.10 and spread[1] <= 0.035)
        if direction == "vertical":
            return bool(spread[1] >= 0.10 and spread[0] <= 0.035)
        centered = points - points.mean(axis=0)
        return bool(spread.min() >= 0.10 and
                    min(np.max(np.abs(centered[:, 0] - centered[:, 1])),
                        np.max(np.abs(centered[:, 0] + centered[:, 1]))) <= 0.045)
    raise ValueError(f"Unsupported task category: {category}")
