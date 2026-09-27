from __future__ import annotations

import argparse

from .ai_runtime import inspect_ai_runtime
from .hardware import detect_hardware
from .planner import build_plan
from .runtime import inspect_torch_runtime


def cmd_doctor(_: argparse.Namespace) -> int:
    hw = detect_hardware()
    print(hw.to_json())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    hw = detect_hardware()
    plan = build_plan(hw, task=args.task, seconds=args.seconds, fps=args.fps)
    print(plan.to_json())
    return 0


def cmd_torch_check(_: argparse.Namespace) -> int:
    report = inspect_torch_runtime(run_smoke_test=True)
    print(report.to_json())
    return 0 if report.cuda_available and report.fp16_smoke_test else 1


def cmd_ai_check(_: argparse.Namespace) -> int:
    report = inspect_ai_runtime()
    print(report.to_json())
    return 0 if report.ready else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zigvideo",
        description="Hardware-aware local AI video generation planner for low-resource PCs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Detect hardware and print capabilities.")
    doctor.set_defaults(func=cmd_doctor)

    torch_check = sub.add_parser(
        "torch-check",
        help="Validate the installed PyTorch CUDA runtime with a small FP16 GPU operation.",
    )
    torch_check.set_defaults(func=cmd_torch_check)

    ai_check = sub.add_parser(
        "ai-check",
        help="Validate Diffusers/LTX/GGUF dependencies before downloading video models.",
    )
    ai_check.set_defaults(func=cmd_ai_check)

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
