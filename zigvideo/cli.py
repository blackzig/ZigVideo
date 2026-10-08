from __future__ import annotations

import argparse
import json

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


def _resolve_generation_backend(requested: str) -> str:
    if requested != "auto":
        return requested

    hw = detect_hardware()
    if (
        hw.cuda_visible
        and 4.0 <= hw.vram_gb <= 6.5
        and (hw.compute_capability or 0) < 8.0
    ):
        return "cogvideox"
    return "ltx"


def cmd_generate(args: argparse.Namespace) -> int:
    backend = _resolve_generation_backend(args.backend)
    print(f"[ZigVideo] Selected backend: {backend}")

    if backend == "cogvideox":
        from .backends.cogvideox_fp16 import (
            generate_text_to_video,
            generation_preview,
        )
    else:
        from .backends.ltx_gguf import (
            generate_text_to_video,
            generation_preview,
        )

    if args.vae_fp32 and backend != "cogvideox":
        parser_error = "--vae-fp32 is currently supported only by the CogVideoX backend."
        raise ValueError(parser_error)

    backend_kwargs = {"vae_fp32": args.vae_fp32} if backend == "cogvideox" else {}

    if args.dry_run:
        print(
            json.dumps(
                generation_preview(
                    args.preset,
                    args.cache_dir,
                    args.output,
                    aspect=args.aspect,
                    num_inference_steps=args.steps,
                    **backend_kwargs,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    report = generate_text_to_video(
        prompt=args.prompt,
        output=args.output,
        preset=args.preset,
        aspect=args.aspect,
        seed=args.seed,
        cache_dir=args.cache_dir,
        num_inference_steps=args.steps,
        **backend_kwargs,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_deliver(args: argparse.Namespace) -> int:
    from .delivery import deliver_video

    report = deliver_video(
        input_path=args.input,
        output_path=args.output,
        target_fps=args.fps,
        target_width=args.width,
        target_height=args.height,
        interpolation=args.interpolation,
    )
    print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zigvideo",
        description="Hardware-aware local AI video generation for low-resource PCs.",
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

    generate = sub.add_parser(
        "generate",
        help="Experimental low-VRAM text-to-video generation.",
    )
    generate.add_argument(
        "--backend",
        choices=["auto", "ltx", "cogvideox"],
        default="auto",
        help="Generation backend. Auto selects CogVideoX on legacy 4-6.5GB Turing/GTX GPUs.",
    )
    generate.add_argument("--prompt", required=True, help="English generation prompt.")
    generate.add_argument(
        "--output",
        default="outputs/first-zigvideo.mp4",
        help="Destination MP4 path.",
    )
    generate.add_argument(
        "--preset",
        choices=["ultra-safe", "safe", "balanced"],
        default="ultra-safe",
        help="Start conservatively on low-VRAM hardware.",
    )
    generate.add_argument(
        "--aspect",
        choices=["16:9", "9:16"],
        default="16:9",
        help="Output composition. Use 9:16 for Shorts, Reels and TikTok.",
    )
    generate.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override diffusion steps for controlled speed/quality experiments.",
    )
    generate.add_argument(
        "--vae-fp32",
        action="store_true",
        help="Diagnostic CogVideoX mode: keep transformer FP16, decode VAE in FP32.",
    )
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument(
        "--cache-dir",
        default="models/huggingface",
        help="Local Hugging Face cache directory.",
    )
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generation/download plan without loading or downloading models.",
    )
    generate.set_defaults(func=cmd_generate)

    deliver = sub.add_parser(
        "deliver",
        help="Convert a generated clip into a social-video delivery format.",
    )
    deliver.add_argument("--input", required=True, help="Source MP4 path.")
    deliver.add_argument("--output", required=True, help="Destination MP4 path.")
    deliver.add_argument("--fps", type=int, default=24)
    deliver.add_argument("--width", type=int, default=720)
    deliver.add_argument("--height", type=int, default=1280)
    deliver.add_argument(
        "--interpolation",
        choices=["motion", "duplicate"],
        default="motion",
        help="Motion interpolation creates intermediate frames; duplicate is faster.",
    )
    deliver.set_defaults(func=cmd_deliver)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
