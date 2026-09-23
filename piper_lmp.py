"""Run the original hierarchical Code as Policies tabletop LMP on Piper MuJoCo."""

from __future__ import annotations

import argparse
import ast
import builtins
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import astunparse
import numpy as np
import shapely.affinity
import shapely.geometry
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer

from piper_demo import MuJoCoTabletop, NOTEBOOK, SCENE
from scene_factory import COLORS, WORKSPACE_MAX, WORKSPACE_MIN


PAPER_PROMPT_COMMIT = "b17226b82733b6d081d2206b8eab26aec0f815d5"
PAPER_PROMPT_DIR = Path(__file__).resolve().parent / "paper_prompts"
PAPER_PROMPT_SHA256 = {
    "tabletop_ui": ("sim_tabletop_ui", "83cdaefd2de3d761d2342ccb5576aa2f903ce77274701afbcaadfbe24b3e3c64"),
    "parse_obj_name": ("sim_parse_obj_name", "f54e71c94d57fefc1510b21deb720552322351a96ad293dca4eea986b3d8b2dd"),
    "parse_position": ("sim_parse_position", "723105b21b0ac237ff785a45bcf7fe65a21ded3361ba28055888dee780ea854f"),
    "fgen": ("fgen", "3b13c457e3b16cf3249bfe8324eaba3476fa019b14b438e5c4e53840d8023178"),
}


def fetch_paper_sim_prompts():
    prompts = {}
    for name, (filename, expected_hash) in PAPER_PROMPT_SHA256.items():
        content = (PAPER_PROMPT_DIR / f"{filename}.txt").read_bytes().removesuffix(b"\n")
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError(f"Official paper prompt checksum mismatch: {filename}")
        prompts[name] = content.decode("utf-8")
    return prompts


SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "isinstance", "len", "list", "map", "max", "min", "print", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
    )
}
NUMPY_ATTRIBUTES = {
    "abs", "all", "any", "arange", "argmax", "argmin", "array", "asarray", "c_",
    "ceil", "clip", "concatenate", "cos", "deg2rad", "diff", "dot", "float32",
    "linspace", "linalg", "max", "mean", "min", "ones", "pi", "r_", "sin",
    "sort", "sqrt", "stack", "sum", "zeros",
}


def execute_generated(code: str, global_vars=None, local_vars=None) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global,
                             ast.Nonlocal, ast.With, ast.AsyncWith)):
            raise ValueError(f"Generated code cannot use {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ValueError("Generated code cannot access private names")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("Generated code cannot access private attributes")
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr not in NUMPY_ATTRIBUTES:
                raise ValueError(f"Generated code cannot use np.{node.attr}")
    variables = dict(global_vars or {})
    variables["__builtins__"] = SAFE_BUILTINS
    locals_dict = local_vars if local_vars is not None else {}
    deadline = time.monotonic() + 120
    previous_trace = sys.gettrace()

    def trace(frame, event, arg):
        if frame.f_code.co_filename == "<lmp-generated>" and time.monotonic() > deadline:
            raise TimeoutError("Generated policy exceeded its execution time limit")
        return trace

    try:
        sys.settrace(trace)
        exec(compile(tree, "<lmp-generated>", "exec"), variables, locals_dict)
    finally:
        sys.settrace(previous_trace)


class CompletionAdapter:
    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.generations = []
        self.Completion = SimpleNamespace(create=self.complete)
        self.Edit = SimpleNamespace(create=self.edit)

    def complete(self, *, prompt, stop, temperature, engine, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Complete the following Python program. "
                 "Return only the continuation in plain Python, without Markdown or the prompt prefix."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("The model returned no Python continuation")
        self.generations.append({
            "model": self.model,
            "prompt": prompt,
            "generated_code": text,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop,
        })
        return {"choices": [{"text": text}]}

    def edit(self, **kwargs):
        raise NotImplementedError("The original bug-fix Edit path is not used by the demo")


def load_original_lmps(client, model: str, protocol: str = "demo"):
    from openai import APIConnectionError, RateLimitError

    notebook = json.loads(NOTEBOOK.read_text())
    namespace = {
        "ast": ast,
        "astunparse": astunparse,
        "copy": copy,
        "np": np,
        "shapely": shapely,
        "highlight": highlight,
        "PythonLexer": PythonLexer,
        "TerminalFormatter": TerminalFormatter,
        "openai": CompletionAdapter(client, model),
        "RateLimitError": RateLimitError,
        "APIConnectionError": APIConnectionError,
        "sleep": time.sleep,
        "model_name": model,
    }
    for cell_index in (5, 14, 15, 16, 17, 18, 19, 21):
        source = "".join(notebook["cells"][cell_index]["source"])
        exec(compile(source, f"original/notebooks/Interactive_Demo.ipynb:cell{cell_index + 1}", "exec"), namespace)
    if protocol == "paper-sim":
        namespace["cfg_tabletop"] = copy.deepcopy(namespace["cfg_tabletop"])
        for name, prompt in fetch_paper_sim_prompts().items():
            namespace["cfg_tabletop"]["lmps"][name]["prompt_text"] = prompt
    elif protocol != "demo":
        raise ValueError(f"Unsupported protocol: {protocol}")
    namespace["exec_safe"] = execute_generated
    return namespace


class MuJoCoLMPWrapper:
    def __init__(self, tabletop: MuJoCoTabletop, min_xy=None, max_xy=None):
        self.tabletop = tabletop
        self.actions = []
        self._min_xy = np.asarray(WORKSPACE_MIN if min_xy is None else min_xy, dtype=float)
        self._max_xy = np.asarray(WORKSPACE_MAX if max_xy is None else max_xy, dtype=float)

    def get_obj_names(self):
        return list(self.tabletop.get_obj_names())

    def is_obj_visible(self, obj_name):
        return obj_name in self.get_obj_names()

    def get_obj_pos(self, obj_name):
        return self.tabletop.get_obj_pos(obj_name)

    def get_bbox(self, obj_name):
        return self.tabletop.get_bbox(obj_name)

    def get_color(self, obj_name):
        if getattr(self.tabletop, "perception", None) is not None:
            return self.tabletop.perception.get_color(obj_name)
        return tuple(channel / 255 for channel in COLORS[obj_name.split()[0]]) + (1.0,)

    def denormalize_xy(self, position):
        return self._min_xy + np.asarray(position) * (self._max_xy - self._min_xy)

    def get_corner_name(self, position):
        corners = np.array([
            [self._min_xy[0], self._max_xy[1]],
            [self._max_xy[0], self._max_xy[1]],
            [self._min_xy[0], self._min_xy[1]],
            [self._max_xy[0], self._min_xy[1]],
        ])
        return ("top left corner", "top right corner", "bottom left corner", "bottom right corner")[
            np.argmin(np.linalg.norm(corners - position, axis=1))
        ]

    def get_side_name(self, position):
        center = (self._max_xy + self._min_xy) / 2
        sides = np.array([
            [center[0], self._max_xy[1]], [self._max_xy[0], center[1]],
            [center[0], self._min_xy[1]], [self._min_xy[0], center[1]],
        ])
        return ("top side", "right side", "bottom side", "left side")[
            np.argmin(np.linalg.norm(sides - position, axis=1))
        ]

    def get_corner_positions(self):
        return np.array([
            [self._min_xy[0], self._min_xy[1]],
            [self._min_xy[0], self._max_xy[1]],
            [self._max_xy[0], self._min_xy[1]],
            [self._max_xy[0], self._max_xy[1]],
        ])

    def get_side_positions(self):
        center = (self._min_xy + self._max_xy) / 2
        return np.array([
            [self._min_xy[0], center[1]], [center[0], self._min_xy[1]],
            [center[0], self._max_xy[1]], [self._max_xy[0], center[1]],
        ])

    def stack_objects_in_order(self, object_names):
        for target, source in zip(object_names, object_names[1:]):
            self.put_first_on_second(source, target)

    def point_gripper_to(self, position):
        raise NotImplementedError("Point-gripper primitive is not implemented for Piper")

    def detect_obj(self, description):
        raise NotImplementedError("Free-form object detection is not implemented")

    def put_first_on_second(self, source, target):
        try:
            self.tabletop.put_first_on_second(source, target)
        except Exception as error:
            self.actions.append((source, target, False))
            print(f"Action result: {source} -> {target}, failed: {error}")
            raise
        source_position = self.tabletop._object_xyz(source)
        target_position = (self.tabletop.get_obj_pos(target) if isinstance(target, str)
                           else np.asarray(target))
        height_ok = (source_position[2] > self.tabletop._object_xyz(target)[2] + 0.036
                     if isinstance(target, str) and target.endswith(" block")
                     else source_position[2] > 0.76)
        success = bool(np.linalg.norm(source_position[:2] - target_position) < 0.03 and height_ok)
        self.actions.append((source, target, success))
        print(f"Action result: {source} -> {target}, success: {success}")


def setup_original_lmp(tabletop: MuJoCoTabletop, client, model: str, protocol: str = "demo",
                       wrapper_cls=MuJoCoLMPWrapper, wrapper_kwargs=None):
    namespace = load_original_lmps(client, model, protocol)
    wrapper = wrapper_cls(tabletop, **(wrapper_kwargs or {}))
    wrapper.generations = namespace["openai"].generations
    fixed_vars = {"np": np}
    for name in shapely.geometry.__all__ + shapely.affinity.__all__:
        fixed_vars[name] = getattr(shapely.geometry, name, None) or getattr(shapely.affinity, name)
    variable_vars = {
        name: getattr(wrapper, name)
        for name in (
            "get_bbox", "get_obj_pos", "get_color", "is_obj_visible", "denormalize_xy",
            "put_first_on_second", "get_obj_names", "get_corner_name", "get_side_name",
            "get_corner_positions", "get_side_positions", "stack_objects_in_order",
            "point_gripper_to", "detect_obj",
        )
    }
    variable_vars["say"] = lambda message: print(f"robot says: {message}")
    cfg = namespace["cfg_tabletop"]["lmps"]
    generator = namespace["LMPFGen"](cfg["fgen"], fixed_vars, variable_vars)
    variable_vars.update({
        name: namespace["LMP"](name, cfg[name], generator, fixed_vars, variable_vars)
        for name in ("parse_obj_name", "parse_position", "parse_question", "transform_shape_pts")
    })
    policy = namespace["LMP"]("tabletop_ui", cfg["tabletop_ui"], generator, fixed_vars, variable_vars)
    return policy, wrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=SCENE)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--protocol", choices=("demo", "paper-sim"), default="demo")
    parser.add_argument("--command", default="put the blue block on the yellow bowl")
    args = parser.parse_args()
    from openai_client import create_openai_client

    client = create_openai_client()
    tabletop = MuJoCoTabletop(args.scene)
    try:
        policy, wrapper = setup_original_lmp(tabletop, client, args.model, args.protocol)
        print(f"Protocol: {args.protocol}; model: {args.model}")
        if args.protocol == "paper-sim":
            print(f"Published prompt commit: {PAPER_PROMPT_COMMIT}")
        initial = {name: tabletop._object_xyz(name) for name in tabletop.get_obj_names()
                   if name.endswith(" block")}
        try:
            policy(args.command, f"objects = {list(tabletop.get_obj_names())}")
        finally:
            final = {name: tabletop._object_xyz(name) for name in initial}
            print(f"Initial block positions: {initial}")
            print(f"Final block positions: {final}")
            print(f"Actions: {wrapper.actions}")
            all_actions_succeeded = bool(wrapper.actions) and all(
                success for _, _, success in wrapper.actions
            )
            print(f"All generated actions succeeded: {all_actions_succeeded}")
    finally:
        tabletop.disconnect()


if __name__ == "__main__":
    main()
