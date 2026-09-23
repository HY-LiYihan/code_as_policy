"""Deterministic language-task instances inspired by Appendix K of arXiv:2209.07753.

This is a template/attribute suite, not the paper's unpublished scene sampler or
benchmark. It records object names and semantic parameters; scene placement,
physics, and success checks belong to a separate environment implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import random


PAPER_URL = "https://arxiv.org/abs/2209.07753"
SEEN_SEEN = "seen_instructions_seen_attributes"
SEEN_UNSEEN = "seen_instructions_unseen_attributes"
UNSEEN_UNSEEN = "unseen_instructions_unseen_attributes"

SEEN_ATTRIBUTES = {
    "colors": ("blue", "red", "green", "orange", "yellow"),
    "sides": ("left", "top-left", "top", "top-right"),
    "directions": ("top", "left"),
    "distances": ("closest",),
    "magnitudes": ("a little",),
    "ordinals": ("first", "second"),
    "lines": ("horizontal", "vertical"),
}
UNSEEN_ATTRIBUTES = {
    "colors": ("pink", "cyan", "brown", "gray", "purple"),
    "sides": ("bottom-right", "bottom", "bottom-left"),
    "directions": ("bottom", "right"),
    "distances": ("farthest",),
    "magnitudes": ("a lot",),
    "ordinals": ("third", "fourth"),
    "lines": ("diagonal",),
}

SEEN_CATEGORIES = (
    "specified_pick_place",
    "stack_all_blocks",
    "all_blocks_to_side",
    "all_blocks_to_bowl",
    "match_colors",
    "relative_to_bowl",
    "distance_to_bowl",
    "nth_from_direction",
)
UNSEEN_CATEGORIES = (
    "different_corners",
    "different_color_bowls",
    "stack_at_side",
    "offset_from_bowl",
    "distance_corner_from_bowl",
    "line_of_blocks",
)
GROUPS = {
    SEEN_SEEN: tuple(f"seen_{index}" for index in range(1, 9)),
    SEEN_UNSEEN: tuple(f"seen_{index}" for index in range(1, 9)),
    UNSEEN_UNSEEN: tuple(f"unseen_{index}" for index in range(1, 7)),
}


@dataclass(frozen=True)
class Case:
    id: str
    split: str
    command: str
    objects: tuple[str, ...]
    seed: int
    task_type: str
    params: dict[str, object]
    goal: dict[str, object]
    metadata: dict[str, object]

    @property
    def category(self) -> str:
        return self.task_type

    @property
    def category_name(self) -> str:
        return str(self.goal["kind"])

    @property
    def parameters(self) -> dict[str, object]:
        return self.params

    @property
    def expected(self) -> dict[str, object]:
        return self.goal

    def to_manifest(self) -> dict[str, object]:
        return json.loads(json.dumps(asdict(self)))


def _derived_seed(seed: int, split: str, category: str, trial: int) -> int:
    digest = sha256(json.dumps([seed, split, category, trial]).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _random_for(category: str, split: str, seed: int) -> random.Random:
    digest = sha256(json.dumps([category, split, seed]).encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _location_label(side: str) -> str:
    if "-" in side:
        return side.replace("-", " ") + " corner"
    return side + " side"


def instantiate(category: str, split: str, seed: int) -> Case:
    """Sample one instruction and scene inventory; never touch global RNG state.

    The seed identifies a trial within a category/split. Objects have no
    positions here: geometric referents must be grounded by a scene sampler.
    """
    if category not in GROUPS.get(split, ()):
        raise ValueError(f"Invalid category/split pair: {category!r}, {split!r}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    rng = _random_for(category, split, seed)
    attrs = SEEN_ATTRIBUTES if split == SEEN_SEEN else UNSEEN_ATTRIBUTES
    choose = rng.choice
    ordinal = choose(attrs["ordinals"]) if category == "seen_8" else None
    block_count = 4 if ordinal == "fourth" else 3
    block_colors = rng.sample(attrs["colors"], block_count)
    bowl_colors = rng.sample(attrs["colors"], 1 if block_count == 4 else 3)
    if category in ("seen_5", "unseen_2"):
        bowl_colors = rng.sample(block_colors, 3)
    blocks = tuple(f"{color} block" for color in block_colors)
    bowls = tuple(f"{color} bowl" for color in bowl_colors)
    parameters: dict[str, object] = {}

    if category == "seen_1":
        source = choose(blocks)
        target = choose(tuple(name for name in blocks if name != source) + bowls)
        parameters.update(source=source, target=target)
        instruction = f"Pick up the {source} and place it on the {target}."
    elif category == "seen_2":
        instruction = "Stack all the blocks."
    elif category == "seen_3":
        side = choose(attrs["sides"])
        parameters["side"] = side
        instruction = f"Put all the blocks on the {_location_label(side)}."
    elif category == "seen_4":
        bowl = choose(bowls)
        parameters["bowl"] = bowl
        instruction = f"Put the blocks in the {bowl}."
    elif category == "seen_5":
        instruction = "Put all the blocks in the bowls with matching colors."
    elif category == "seen_6":
        bowl, direction, side = choose(bowls), choose(attrs["directions"]), choose(attrs["sides"])
        parameters.update(bowl=bowl, direction=direction, side=side)
        instruction = (f"Pick up the block to the {direction} of the {bowl} "
                       f"and place it on the {_location_label(side)}.")
    elif category == "seen_7":
        bowl, distance, side = choose(bowls), choose(attrs["distances"]), choose(attrs["sides"])
        parameters.update(bowl=bowl, distance=distance, side=side)
        instruction = (f"Pick up the block {distance} to the {bowl} "
                       f"and place it on the {_location_label(side)}.")
    elif category == "seen_8":
        direction, side = choose(attrs["directions"]), choose(attrs["sides"])
        parameters.update(ordinal=ordinal, direction=direction, side=side)
        instruction = (f"Pick up the {ordinal} block from the {direction} "
                       f"and place it on the {_location_label(side)}.")
    elif category == "unseen_1":
        instruction = "Put all the blocks in different corners."
    elif category == "unseen_2":
        instruction = "Put the blocks in the bowls with mismatched colors."
    elif category == "unseen_3":
        side = choose(attrs["sides"])
        parameters["side"] = side
        instruction = f"Stack all the blocks on the {_location_label(side)}."
    elif category == "unseen_4":
        block, bowl = choose(blocks), choose(bowls)
        direction, magnitude = choose(attrs["directions"]), choose(attrs["magnitudes"])
        parameters.update(block=block, bowl=bowl, direction=direction, magnitude=magnitude)
        instruction = f"Pick up the {block} and place it {magnitude} to the {direction} of the {bowl}."
    elif category == "unseen_5":
        block, bowl, distance = choose(blocks), choose(bowls), choose(attrs["distances"])
        parameters.update(block=block, bowl=bowl, distance=distance)
        instruction = f"Pick up the {block} and place it in the corner {distance} to the {bowl}."
    else:
        line = choose(attrs["lines"])
        parameters["line"] = line
        instruction = f"Put all the blocks in a {line} line."

    names = blocks + bowls
    if category.startswith("seen_"):
        category_name = SEEN_CATEGORIES[int(category.split("_")[1]) - 1]
    else:
        category_name = UNSEEN_CATEGORIES[int(category.split("_")[1]) - 1]
    goal: dict[str, object] = {
        "kind": category_name,
        "parameters": parameters.copy(),
        "blocks": blocks,
        "bowls": bowls,
    }
    if category == "unseen_1":
        goal["constraint"] = "distinct_corners"
    elif category == "unseen_2":
        goal["constraint"] = "each_block_in_a_different_color_bowl"
    elif category in ("seen_5", "seen_6", "seen_7", "seen_8", "unseen_5"):
        goal["constraint"] = "resolve_referents_from_initial_positions" if category in ("seen_6", "seen_7", "seen_8", "unseen_5") else "match_block_and_bowl_colors"
    if category == "seen_6":
        goal["precondition"] = "at_least_one_block_in_specified_direction_of_bowl"
    return Case(
        id=f"{split}/{category}/{seed}", split=split, command=instruction,
        objects=names, seed=seed, task_type=category, params=parameters,
        goal=goal, metadata={"source": PAPER_URL, "category_name": category_name},
    )


def generate_cases(seed: int, repetitions: int = 1) -> list[Case]:
    """Generate one independently seeded case per category/split and repetition.

    Pass each case's ``objects`` and ``seed`` to the scene factory. Its sampled
    positions must satisfy any ``goal['precondition']`` before evaluation;
    this module deliberately does not import or call the scene factory.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    cases = []
    for split, categories in GROUPS.items():
        for category in categories:
            for trial in range(repetitions):
                case_seed = _derived_seed(seed, split, category, trial)
                case = instantiate(category, split, case_seed)
                cases.append(Case(
                    id=f"{seed}/{split}/{category}/{trial:04d}", split=case.split,
                    command=case.command, objects=case.objects, seed=case.seed,
                    task_type=case.task_type, params=case.params, goal=case.goal,
                    metadata=case.metadata,
                ))
    return cases


def build_manifest(seed: int, trials_per_category: int = 50) -> dict[str, object]:
    """Return 22 independently labeled category/split cells as JSON-ready data."""
    cases = generate_cases(seed, trials_per_category)
    groups: dict[str, dict[str, list[dict[str, object]]]] = {
        split: {category: [] for category in categories} for split, categories in GROUPS.items()
    }
    for case in cases:
        groups[case.split][case.category].append(case.to_manifest())
    return {
        "source": PAPER_URL,
        "scope": "Appendix K-inspired templates and attributes; not the original hidden instance distribution or benchmark",
        "seed": seed,
        "trials_per_category": trials_per_category,
        "groups": groups,
    }
