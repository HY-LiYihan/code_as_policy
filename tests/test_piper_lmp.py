import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from piper_lmp import (MuJoCoLMPWrapper, execute_generated, fetch_paper_sim_prompts,
                       load_original_lmps, setup_original_lmp)


class FakeTabletop:
    initial_positions = {
        "blue block": np.array([-0.05, -0.12]),
        "green block": np.array([-0.13, 0.04]),
        "red block": np.array([-0.05, 0.16]),
        "yellow bowl": np.array([0.05, 0.08]),
    }

    def __init__(self):
        self.positions = {name: position.copy() for name, position in self.initial_positions.items()}
        self.movements = []

    def get_obj_names(self):
        return tuple(self.positions)

    def get_obj_pos(self, name):
        return self.positions[name]

    def get_bbox(self, name):
        return (0.0, 0.0, 0.1, 0.1)

    def put_first_on_second(self, source, target):
        self.movements.append((source, target))
        self.positions[source] = self.positions[target].copy()

    def _object_xyz(self, name):
        return np.r_[self.positions[name], 0.78]


class FakeCompletions:
    def __init__(self):
        self.responses = iter([
            "block_name = parse_obj_name('the block closest to the yellow bowl', "
            "f'objects = {get_obj_names()}')\n"
            "put_first_on_second(block_name, 'yellow bowl')",
            "block_names = ['blue block', 'green block', 'red block']\n"
            "positions = get_obj_positions_np(block_names)\n"
            "index = get_closest_idx(points=positions, point=get_obj_pos('yellow bowl'))\n"
            "ret_val = block_names[index]",
            "def get_obj_positions_np(block_names):\n"
            "    return np.array([get_obj_pos(name) for name in block_names])",
            "def get_closest_idx(points, point):\n"
            "    return np.argmin(np.linalg.norm(points - point, axis=1))",
        ])
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = next(self.responses)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class OriginalLMPTests(unittest.TestCase):
    def test_paper_sim_uses_pinned_official_prompts(self):
        content = b"# published paper simulation prompt\n"
        with (TemporaryDirectory() as temporary,
              patch("piper_lmp.PAPER_PROMPT_DIR", Path(temporary)),
              patch("piper_lmp.PAPER_PROMPT_SHA256", {"tabletop_ui": ("sim_tabletop_ui", sha256(content).hexdigest())})):
            (Path(temporary) / "sim_tabletop_ui.txt").write_bytes(content + b"\n")
            prompts = fetch_paper_sim_prompts()
            namespace = load_original_lmps(None, "test-model", "paper-sim")
            self.assertEqual(prompts["tabletop_ui"], content.decode())
            self.assertEqual(namespace["cfg_tabletop"]["lmps"]["tabletop_ui"]["prompt_text"], content.decode())

    def test_paper_sim_rejects_changed_prompt(self):
        with (TemporaryDirectory() as temporary,
              patch("piper_lmp.PAPER_PROMPT_DIR", Path(temporary)),
              patch("piper_lmp.PAPER_PROMPT_SHA256", {"tabletop_ui": ("sim_tabletop_ui", "0" * 64)})):
            (Path(temporary) / "sim_tabletop_ui.txt").write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                fetch_paper_sim_prompts()

    def test_checked_in_paper_prompts_match_published_hashes(self):
        self.assertEqual(len(fetch_paper_sim_prompts()), 4)

    def test_stack_uses_sequential_pick_place(self):
        tabletop = FakeTabletop()
        wrapper = MuJoCoLMPWrapper(tabletop)
        wrapper.stack_objects_in_order(["yellow bowl", "blue block", "red block"])
        self.assertEqual(tabletop.movements, [("blue block", "yellow bowl"), ("red block", "blue block")])

    def test_original_hierarchical_generation_and_execution(self):
        tabletop = FakeTabletop()
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        policy, wrapper = setup_original_lmp(tabletop, client, "gpt-6-astra")
        policy("put the block closest to the yellow bowl on the yellow bowl",
               f"objects = {list(tabletop.get_obj_names())}")
        self.assertEqual(len(completions.calls), 4)
        self.assertEqual(tabletop.movements, [("red block", "yellow bowl")])
        self.assertEqual(wrapper.actions, [("red block", "yellow bowl", True)])

    def test_paper_sim_prompt_runs_hierarchical_generation(self):
        tabletop = FakeTabletop()
        completions = FakeCompletions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        policy, wrapper = setup_original_lmp(tabletop, client, "test-model", "paper-sim")
        policy("put the block closest to the yellow bowl on the yellow bowl",
               f"objects = {list(tabletop.get_obj_names())}")
        self.assertIn("from env_utils import get_obj_pos, detect_obj", completions.calls[0]["messages"][1]["content"])
        self.assertEqual(len(completions.calls), 4)
        self.assertEqual(wrapper.actions, [("red block", "yellow bowl", True)])

    def test_generated_code_cannot_import_or_access_private_attributes(self):
        for code in ("import os", "np.load('/tmp/data')", "obj.__class__"):
            with self.subTest(code=code), self.assertRaises(ValueError):
                execute_generated(code, {"np": np})


if __name__ == "__main__":
    unittest.main()
