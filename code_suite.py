"""Independent task cases drawn from the released Interactive_Demo notebook examples."""

from __future__ import annotations

from hashlib import sha256

from paper_suite import Case


EXAMPLES = (
    ("demo_corner", "Move the green block to the top right corner.",
     ("yellow block", "green block", "blue block", "yellow bowl", "blue bowl", "green bowl")),
    ("demo_leftmost", "Move the left most block to the green bowl.",
     ("pink block", "green block", "blue block", "pink bowl", "blue bowl", "green bowl")),
    ("demo_farthest", "Put the red block on the farthest bowl.",
     ("red block", "brown block", "pink block", "brown bowl", "red bowl", "pink bowl")),
    ("demo_bottom", "Move the pinkish colored block on the bottom side.",
     ("pink block", "green block", "orange block")),
    ("demo_stack", "Stack the blocks on the right side with the gray one on the bottom.",
     ("yellow block", "green block", "gray block", "yellow bowl", "gray bowl", "green bowl")),
)


def generate_code_cases(seed: int, repetitions: int = 1) -> list[Case]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    cases = []
    for category, command, objects in EXAMPLES:
        blocks = tuple(name for name in objects if name.endswith(" block"))
        bowls = tuple(name for name in objects if name.endswith(" bowl"))
        for trial in range(repetitions):
            digest = sha256(f"{seed}:{category}:{trial}".encode()).digest()
            case_seed = int.from_bytes(digest[:8], "big")
            cases.append(Case(
                id=f"{seed}/notebook_demo/{category}/{trial:04d}",
                split="notebook_demo", command=command, objects=objects, seed=case_seed,
                task_type=category, params={},
                goal={"kind": category, "parameters": {}, "blocks": blocks, "bowls": bowls},
                metadata={"source": "original/notebooks/Interactive_Demo.ipynb",
                          "note": "one-shot notebook example"},
            ))
    return cases
