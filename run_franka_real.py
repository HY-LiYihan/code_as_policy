"""Run the original Code as Policies tabletop LMP against a real Franka FR3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from franka_real import load_observations, plan_from_observations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-json", type=Path, required=True,
                        help="Object observations in the Franka base frame")
    parser.add_argument("--command", required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--robot-ip", default=None)
    parser.add_argument("--motion-duration", type=float, default=None)
    parser.add_argument("--table-z", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true",
                        help="Send the generated pick-and-place to the real robot")
    parser.add_argument("--confirm-real", action="store_true",
                        help="Required together with --execute")
    args = parser.parse_args()
    if args.execute and not args.confirm_real:
        parser.error("--execute requires --confirm-real")
    from openai_client import create_openai_client

    try:
        client = create_openai_client()
    except ValueError as exc:
        parser.error(str(exc))

    observations = load_observations(args.objects_json)
    tabletop = None
    try:
        tabletop, wrapper = plan_from_observations(
            observations, args.command, client=client, model=args.model,
            execute=args.execute, robot_ip=args.robot_ip,
            motion_duration_s=args.motion_duration, table_z=args.table_z,
        )
        print(json.dumps({
            "mode": "execute" if args.execute else "dry-run",
            "command": args.command,
            "generations": wrapper.generations,
            "actions": wrapper.actions,
        }, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
    finally:
        if tabletop is not None:
            tabletop.disconnect()


if __name__ == "__main__":
    main()
