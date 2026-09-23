"""Run frozen Code as Policies tabletop cases in Piper MuJoCo, or plan them offline."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import numpy as np
from robot_control.api.types import Pose

from code_suite import EXAMPLES, generate_code_cases
from paper_suite import GROUPS, generate_cases
from piper_demo import DOWN_QUATERNION, SCENE, MuJoCoTabletop
from piper_lmp import PAPER_PROMPT_COMMIT, PAPER_PROMPT_SHA256, setup_original_lmp
from repro_metadata import (build_manifest, dump_json, wilson_interval,
                            workspace_calibration_metadata)
from scene_factory import FORWARD_OFFSET_METERS, sample_positions, write_scene
from task_scoring import scene_precondition, score_case


def to_json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def downward_ik_precondition(case, positions, backend):
    heights = (0.92, 0.93, 0.965, 0.97)
    for xy in positions.values():
        for height in heights:
            pose = backend._transform_ik_pose(Pose((*xy, height), DOWN_QUATERNION), inverse=True)
            solution = backend.ik.solve(
                pose, seed=backend.data.qpos[backend._arm_qpos].copy())
            if not solution.success:
                return False
    return True


def plan_positions(case, ik_backend=None):
    for attempt in range(128):
        scene_seed = (case.seed + attempt) % (2 ** 64)
        positions = sample_positions(case.objects, scene_seed)
        if scene_precondition(case, positions) and (ik_backend is None or
                                                    downward_ik_precondition(case, positions, ik_backend)):
            return positions, scene_seed, attempt
    raise ValueError(f"Could not sample a valid task scene with downward IK for {case.id}")


def record_trial(case, scene: Path, bodies: dict[str, str], client, model: str, protocol: str,
                 video_path: Path | None = None, visual_sam3: bool = False,
                 sam3_host: str = "127.0.0.1", sam3_port: int = 28317):
    tabletop = MuJoCoTabletop(scene, object_bodies=bodies)
    record = {"initial_xyz": {}, "final_xyz": {}, "generations": [], "actions": [],
              "error": None, "phase": "setup", "action_success": False, "task_success": None}
    start = time.monotonic()
    wrapper = None
    video = None
    try:
        if video_path is not None:
            from trial_video import TrialVideo

            video = TrialVideo(tabletop.backend, video_path)
            record["video"] = str(video_path)
        if visual_sam3:
            tabletop.attach_wrist_sam3_perception(
                texts=list(bodies), host=sam3_host, port=sam3_port)
            observations = tabletop.perception.refresh()
            record["perception"] = {
                "mode": "sam3_rgbd",
                "camera": "wrist",
                "observations": {
                    name: {"xyz": observation.xyz, "bbox_xy": observation.bbox_xy,
                           "bbox_pixels": observation.bbox_pixels,
                           "color_rgb": observation.color_rgb}
                    for name, observation in observations.items()
                },
            }
        record["initial_xyz"] = {name: tabletop._object_xyz(name) for name in bodies}
        policy, wrapper = setup_original_lmp(tabletop, client, model, protocol=protocol)
        record["phase"] = "generation"
        try:
            policy(case.command.rstrip("."), f"objects = {list(bodies)}")
        except Exception as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        finally:
            record["generations"] = wrapper.generations
            record["actions"] = wrapper.actions
            record["final_xyz"] = {name: tabletop._object_xyz(name) for name in bodies}
            record["action_success"] = bool(wrapper.actions) and all(action[2] for action in wrapper.actions)
            record["task_success"] = bool(record["action_success"] and record["error"] is None
                                          and score_case(case, record["initial_xyz"], record["final_xyz"]))
            if record["error"] is not None:
                record["phase"] = ("execution" if wrapper.actions else
                                   "policy" if wrapper.generations else "generation")
            else:
                record["phase"] = "completed"
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        try:
            if video is not None:
                video.close()
                record["video_frames"] = video.frames
                record["video_seconds"] = video.frames / video.fps
                record["video_simulation_seconds"] = video.simulation_seconds
        finally:
            tabletop.disconnect()
    record["failure_stage"] = classify_failure(record)
    return record


def classify_failure(record: dict) -> str | None:
    if record.get("task_success"):
        return None
    error = record.get("error") or {}
    error_type = error.get("type", "")
    message = error.get("message", "").lower()
    phase = record.get("phase")
    if error_type == "IKError" or "ik failed" in message:
        return "ik"
    if phase in ("policy", "generation"):
        return "generation"
    if "grasp" in message or "contact" in message or "stay on top" in message:
        return "grasp"
    if "deposit" in message or "target" in message or "release" in message:
        return "placement"
    if phase == "completed" or (record.get("action_success") and not error):
        return "scoring"
    if phase == "execution":
        return "execution"
    return "setup"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("paper", "code-demo"), default="paper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=tuple(GROUPS) + ("notebook_demo",))
    parser.add_argument("--category", choices=tuple(dict.fromkeys(
        [category for categories in GROUPS.values() for category in categories]
        + [example[0] for example in EXAMPLES])))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--output-dir", type=Path, default=Path("results/canonical/paper-sim"))
    parser.add_argument("--record-video", action="store_true", help="Record real simulation steps to a third-person MP4")
    parser.add_argument("--record-representative-videos", action="store_true",
                        help="Keep the first successful and first failed video per experiment group")
    parser.add_argument("--require-downward-ik", action="store_true",
                        help="Select the first deterministic seen_1 scene with IK for all downward waypoints")
    parser.add_argument("--blocks-only", action="store_true",
                        help="Generate and perceive only block objects; omit all bowls from the scene")
    parser.add_argument("--visual-sam3", action="store_true",
                        help="Use wrist RGB-D plus the SAM3 TCP service for object queries")
    parser.add_argument("--sam3-host", default=os.environ.get("SAM3_HOST", "127.0.0.1"))
    parser.add_argument("--sam3-port", type=int, default=int(os.environ.get("SAM3_PORT", "28317")))
    args = parser.parse_args()
    if args.trials < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--trials and --limit must be positive")
    if args.plan_only and args.record_video:
        parser.error("--record-video requires an online run")
    if args.plan_only and args.record_representative_videos:
        parser.error("--record-representative-videos requires an online run")
    if args.require_downward_ik and args.suite != "paper":
        parser.error("--require-downward-ik requires --suite paper")
    if args.blocks_only and (args.suite != "paper" or args.category != "seen_1"):
        parser.error("--blocks-only currently requires --suite paper --category seen_1")
    cases = (generate_cases(seed=args.seed, repetitions=args.trials) if args.suite == "paper"
             else generate_code_cases(seed=args.seed, repetitions=args.trials))
    if args.split is not None:
        cases = [case for case in cases if case.split == args.split]
    if args.category is not None:
        cases = [case for case in cases if case.task_type == args.category]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        parser.error("No cases match this split and category")
    output = args.output_dir / "trials.jsonl"
    if output.exists():
        parser.error(f"Refusing to overwrite existing results: {output}")
    client = None
    if not args.plan_only:
        from openai_client import create_openai_client

        try:
            client = create_openai_client()
        except ValueError as exc:
            parser.error(str(exc) + ", or use --plan-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    representative_videos = {}
    protocol = "paper-sim" if args.suite == "paper" else "demo"
    prompt_sha256 = ({name: digest for name, (_, digest) in PAPER_PROMPT_SHA256.items()}
                     if args.suite == "paper" else {})
    scene_protocol = {
        "forward_offset_m": FORWARD_OFFSET_METERS,
        "placement_jitter_m": 0.008,
        "minimum_initial_separation_m": 0.15,
        "workspace_min_xy": [-0.19 + FORWARD_OFFSET_METERS, -0.19],
        "workspace_max_xy": [0.01 + FORWARD_OFFSET_METERS, 0.19],
        "workspace_calibration": workspace_calibration_metadata(),
    }
    manifest = build_manifest(
        suite=args.suite, protocol=protocol, model=None if args.plan_only else args.model,
        seed=args.seed, trials=args.trials, visual_sam3=args.visual_sam3,
        require_downward_ik=args.require_downward_ik, blocks_only=args.blocks_only,
        prompt_commit=PAPER_PROMPT_COMMIT if args.suite == "paper" else None,
        prompt_sha256=prompt_sha256,
        scene_protocol=scene_protocol,
    )
    dump_json(args.output_dir / "manifest.json", manifest)
    ik_tabletop = MuJoCoTabletop(SCENE) if args.require_downward_ik else None
    with output.open("x", encoding="utf-8") as results:
        for index, case in enumerate(cases):
            scene = args.output_dir / "scenes" / f"trial_{index:04d}.xml"
            record = {
                "case": dataclasses.asdict(case), "scene": str(scene),
                "suite": args.suite, "protocol": protocol,
                "scene_forward_offset_m": FORWARD_OFFSET_METERS,
                "downward_ik_preflight": args.require_downward_ik,
                "model": None if args.plan_only else args.model,
                "requested_model": args.model,
                "published_prompt_commit": PAPER_PROMPT_COMMIT if args.suite == "paper" else None,
                "prompt_sha256": ({name: digest for name, (_, digest) in PAPER_PROMPT_SHA256.items()}
                                  if args.suite == "paper" else {}),
                "implementation_sha256": manifest["implementation_sha256"],
                "backend_commit": manifest["backend"]["commit"],
            }
            try:
                positions, scene_seed, attempts = plan_positions(
                    case, ik_backend=ik_tabletop.backend if ik_tabletop is not None else None)
                if args.blocks_only:
                    positions = {name: positions[name] for name in case.objects
                                 if name.endswith(" block")}
                bodies = write_scene(positions, scene)
                record.update(object_bodies=bodies, sampled_xy=positions,
                              scene_seed=scene_seed, sampling_attempts=attempts,
                              blocks_only=args.blocks_only)
                if args.plan_only:
                    record["phase"] = "planned"
                else:
                    video_path = None
                    group_key = case.split
                    if args.record_video:
                        video_path = args.output_dir / "videos" / f"trial_{index:04d}.mp4"
                    elif args.record_representative_videos:
                        statuses = representative_videos.setdefault(group_key, set())
                        if len(statuses) < 2:
                            video_path = args.output_dir / "videos" / f".candidate_{index:04d}.mp4"
                    record.update(record_trial(case, scene, bodies, client, args.model, protocol,
                                               video_path=video_path, visual_sam3=args.visual_sam3,
                                               sam3_host=args.sam3_host, sam3_port=args.sam3_port))
                    if args.record_representative_videos and video_path is not None:
                        status = "success" if record.get("task_success") else "failure"
                        statuses = representative_videos.setdefault(case.split, set())
                        if status not in statuses and video_path.exists():
                            final_video = args.output_dir / "videos" / (
                                f"{case.split}-{status}-trial-{index:04d}.mp4")
                            video_path.rename(final_video)
                            record["video"] = str(final_video)
                            statuses.add(status)
                        elif video_path.exists():
                            video_path.unlink()
                            record.pop("video", None)
            except Exception as error:
                record["phase"] = "setup"
                record["error"] = {"type": type(error).__name__, "message": str(error)}
                record["task_success"] = False if not args.plan_only else None
                record["failure_stage"] = classify_failure(record)
            results.write(json.dumps(record, ensure_ascii=False, default=to_json) + "\n")
            results.flush()
            completed.append(record)
            print(f"{index + 1}/{len(cases)} {case.id}: {record['phase']}, "
                  f"task_success={record.get('task_success')}")
    groups = {}
    for record in completed:
        case = record["case"]
        cell = groups.setdefault(case["split"], {}).setdefault(case["task_type"],
                                                                  {"trials": 0, "successes": 0, "failures": 0,
                                                                   "failure_stages": {}})
        cell["trials"] += 1
        cell["successes" if record.get("task_success") else "failures"] += int(not args.plan_only)
        if not args.plan_only and record.get("failure_stage"):
            stage = record["failure_stage"]
            cell["failure_stages"][stage] = cell["failure_stages"].get(stage, 0) + 1
    for categories in groups.values():
        for cell in categories.values():
            cell["success_rate"] = None if args.plan_only else cell["successes"] / cell["trials"]
            cell["success_rate_ci95"] = None if args.plan_only else wilson_interval(
                cell["successes"], cell["trials"])
    dump_json(args.output_dir / "summary.json", {
        "suite": args.suite, "protocol": protocol, "planned_only": args.plan_only,
        "scene_forward_offset_m": FORWARD_OFFSET_METERS,
        "downward_ik_preflight": args.require_downward_ik,
        "model": None if args.plan_only else args.model,
        "requested_model": args.model, "seed": args.seed,
        "trials_per_category_requested": args.trials, "case_count": len(completed), "groups": groups,
        "implementation_sha256": manifest["implementation_sha256"],
        "backend_commit": manifest["backend"]["commit"],
        "manifest": "manifest.json",
        "representative_videos": {
            group: sorted(statuses) for group, statuses in representative_videos.items()
        },
        "phases": {phase: sum(record.get("phase") == phase for record in completed)
                   for phase in sorted({record.get("phase") for record in completed})},
    })
    print(f"Trial records: {output}")
    if ik_tabletop is not None:
        ik_tabletop.disconnect()


if __name__ == "__main__":
    main()
