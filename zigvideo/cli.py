from __future__ import annotations

import argparse

from .hardware import detect_hardware
from .planner import build_plan


def cmd_doctor(_: argparse.Namespace) -> int:
    hw = detect_hardware()
    print(hw.to_json())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    hw = detect_hardware()
    plan = build_plan(hw, task=args.task, seconds=args.seconds, fps=args.fps)
    print(plan.to_json())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zigvideo",
        description="Hardware-aware local AI video generation planner for low-resource PCs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Detect hardware and print capabilities.")
    doctor.set_defaults(func=cmd_doctor)

    plan = sub.add_parser("plan", help="Create a low-VRAM generation plan for this PC.")
    plan.add_argument("--task", choices=["t2v", "i2v"], default="i2v")
    plan.add_argument("--seconds", type=float, default=5.0)
    plan.add_argument("--fps", type=int, default=16)
    plan.set_defaults(func=cmd_plan)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
