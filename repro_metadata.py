"""Metadata and statistics helpers for auditable reproduction runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_BACKEND_ROOT = ROOT / "robot_control"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_hashes() -> dict[str, str]:
    files = (
        "code_suite.py", "paper_suite.py", "scene_factory.py", "task_scoring.py",
        "piper_lmp.py", "piper_demo.py", "rgbd_perception.py", "sam3_client.py",
        "trial_video.py", "run_paper_suite.py", "repro_metadata.py", "workspace_calibration.py",
        "franka_real.py", "run_franka_real.py", "openai_client.py",
        "rgbd_perception.py",
        "scenes/piper_tabletop.xml",
    )
    return {name: sha256_file(ROOT / name) for name in files}


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], text=True).strip()


def backend_metadata(repository: Path = DEFAULT_BACKEND_ROOT) -> dict:
    if not (repository / ".git").exists():
        raise FileNotFoundError("robot_control submodule is missing; run git submodule update --init --recursive")
    status = _git(repository, "status", "--porcelain")
    if status:
        raise RuntimeError("robot_control working tree must be clean before a canonical run")
    commit = _git(repository, "rev-parse", "HEAD")
    if repository.resolve() == DEFAULT_BACKEND_ROOT.resolve():
        gitlink = _git(ROOT, "ls-files", "--stage", "--", "robot_control").split()
        if len(gitlink) != 4 or gitlink[0] != "160000" or gitlink[1] != commit:
            raise RuntimeError("robot_control checkout must match the Git submodule commit")
    submodules = _git(repository, "submodule", "status", "--recursive")
    if any(line.startswith(("-", "+", "U")) for line in submodules.splitlines()):
        raise RuntimeError("robot_control nested submodules must match the pinned commits; "
                           "run git submodule update --init --recursive")
    return {
        "repository": "https://github.com/HY-LiYihan/robot_control",
        "commit": commit,
        "branch": _git(repository, "branch", "--show-current") or "detached",
        "submodules": submodules.splitlines() if submodules else [],
    }


def package_versions() -> dict[str, str]:
    names = ("robot-control", "mujoco", "pin", "numpy", "openai", "astunparse", "shapely")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def build_manifest(*, suite: str, protocol: str, model: str | None,
                   seed: int, trials: int, visual_sam3: bool,
                   require_downward_ik: bool, blocks_only: bool,
                   prompt_commit: str | None, prompt_sha256: dict[str, str],
                   scene_protocol: dict, backend_root: Path = DEFAULT_BACKEND_ROOT) -> dict:
    backend = backend_metadata(backend_root)
    calibration = scene_protocol.get("workspace_calibration")
    if calibration is not None and calibration["backend_commit"] != backend["commit"]:
        raise ValueError("Workspace calibration backend differs from the current robot_control commit; "
                         "rerun workspace_calibration.py before starting a new experiment")
    return {
        "manifest_version": 1,
        "suite": suite,
        "protocol": protocol,
        "provider": "OpenAI",
        "model": model,
        "seed": seed,
        "trials_per_category_requested": trials,
        "visual_sam3": visual_sam3,
        "sam3": {"enabled": visual_sam3} if visual_sam3 else None,
        "downward_ik_preflight": require_downward_ik,
        "blocks_only": blocks_only,
        "published_prompt_commit": prompt_commit,
        "prompt_sha256": prompt_sha256,
        "scene_protocol": scene_protocol,
        "implementation_sha256": implementation_hashes(),
        "backend": backend,
        "python": {"version": sys.version},
        "packages": package_versions(),
    }


def wilson_interval(successes: int, trials: int, confidence_z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    rate = successes / trials
    denominator = 1 + confidence_z**2 / trials
    center = (rate + confidence_z**2 / (2 * trials)) / denominator
    radius = confidence_z * (rate * (1 - rate) / trials + confidence_z**2 / (4 * trials**2)) ** 0.5 / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def dump_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def workspace_calibration_metadata() -> dict | None:
    path = ROOT / "results" / "canonical" / "workspace-calibration.json"
    if not path.is_file():
        return None
    calibration = json.loads(path.read_text())
    return {
        "path": "results/canonical/workspace-calibration.json",
        "sha256": sha256_file(path),
        "grid_points": calibration["grid_points"],
        "reachable_points": calibration["reachable_points"],
        "step_m": calibration["step_m"],
        "heights_m": calibration["heights_m"],
        "reachable_min_xy": calibration["reachable_min_xy"],
        "reachable_max_xy": calibration["reachable_max_xy"],
        "backend_commit": calibration["backend"]["commit"],
    }
