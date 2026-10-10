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

    if (
        args.vae_fp32
        or args.save_latents
        or args.latents_only
        or args.cfg_scale is not None
        or args.cfg_guided_steps is not None
        or args.cfg_guided_start != 0
        or args.experimental_cfg_video
        or args.offload != "sequential"
    ) and backend != "cogvideox":
        raise ValueError(
            "--vae-fp32, --save-latents, --latents-only, --cfg-scale, "
            "--cfg-guided-steps, --experimental-cfg-video "
            "and group offloading require the CogVideoX backend."
        )

    backend_kwargs = (
        {
            "vae_fp32": args.vae_fp32,
            "save_latents": args.save_latents,
            "offload_strategy": args.offload,
            "latents_only": args.latents_only,
            "cfg_scale": args.cfg_scale,
            "cfg_guided_steps": args.cfg_guided_steps,
            "cfg_guided_start": args.cfg_guided_start,
            "experimental_cfg_video": args.experimental_cfg_video,
        }
        if backend == "cogvideox"
        else {}
    )

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


def cmd_decode(args: argparse.Namespace) -> int:
    from .decode import decode_saved_latents, inspect_latent_file
    from .backends.cogvideox_fp16 import validate_reframe_focus

    validate_reframe_focus(args.aspect, args.focus_x)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": "cogvideox-vae-only",
                    "source": inspect_latent_file(args.latents),
                    "vae_precision": "float32",
                    "device": args.device,
                    "offload_strategy": ("leaf_level" if args.device == "cuda" else "none"),
                    "aspect": args.aspect,
                    "focus_x": args.focus_x,
                    "fps": args.fps,
                    "output": args.output,
                    "cache_dir": args.cache_dir,
                    "note": "VAE only, no transformer or diffusion. CUDA uses experimental leaf-level CPU offload; dry-run does not check available VRAM.",
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    report = decode_saved_latents(
        input_latents=args.latents,
        output=args.output,
        aspect=args.aspect,
        fps=args.fps,
        cache_dir=args.cache_dir,
        device=args.device,
        focus_x=args.focus_x,
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
        "--offload",
        choices=["sequential", "group"],
        default="sequential",
        help=(
            "CogVideoX CPU/GPU transfer strategy. 'group' is experimental; "
            "use --steps 1 for an OOM/speed smoke test first."
        ),
    )
    generate.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help=(
            "Experimental CogVideoX CFG override. Baseline is 6; CFG 1 "
            "avoids double-batch guidance but may change visual quality."
        ),
    )
    generate.add_argument(
        "--cfg-guided-steps",
        type=int,
        default=None,
        help=(
            "Experimental CogVideoX sequential-only schedule: use normal CFG "
            "for the first N steps and conditional-only transformer computation "
            "thereafter. Requires --latents-only or explicit "
            "--experimental-cfg-video, and CFG > 1."
        ),
    )
    generate.add_argument(
        "--cfg-guided-start",
        type=int,
        default=0,
        help=(
            "Zero-based starting step for selective CFG (default 0). "
            "Requires --cfg-guided-steps; together they define a "
            "guided window instead of always starting at the first step."
        ),
    )
    generate.add_argument(
        "--experimental-cfg-video",
        action="store_true",
        help=(
            "Explicitly opt into a visually unvalidated selective-CFG MP4 "
            "generation. Requires --cfg-guided-steps and sequential offload; "
            "automatically saves final latents before VAE decoding."
        ),
    )
    generate.add_argument(
        "--latents-only",
        action="store_true",
        help=(
            "CogVideoX diffusion-only benchmark: save final .latents.safetensors "
            "and JSON, skipping VAE decoding and MP4 export."
        ),
    )
    generate.add_argument(
        "--vae-fp32",
        action="store_true",
        help="Diagnostic CogVideoX mode: keep transformer FP16, decode VAE in FP32.",
    )
    generate.add_argument(
        "--save-latents",
        action="store_true",
        help="Save the final CogVideoX latents to a safetensors file for decoder debugging.",
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

    decode = sub.add_parser(
        "decode",
        help="Decode saved CogVideoX latents using only an FP32 VAE (CPU default, CUDA experimental).",
    )
    decode.add_argument(
        "--latents", required=True, help="Saved .latents.safetensors input."
    )
    decode.add_argument(
        "--output", required=True, help="Destination MP4 file."
    )
    decode.add_argument("--fps", type=int, default=8)
    decode.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu",
        help="FP32 VAE decode target. CUDA uses leaf-level CPU offload and is experimental.",
    )
    decode.add_argument("--aspect", choices=["9:16", "16:9"], default="9:16")
    decode.add_argument(
        "--focus-x", type=float, default=0.5,
        help=(
            "Horizontal subject center as a fraction of native width for "
            "portrait crops: 0=left, 0.5=center (unchanged default), "
            "1=right. Only applies with --aspect 9:16."
        ),
    )
    decode.add_argument(
        "--cache-dir", default="models/huggingface"
    )
    decode.add_argument(
        "--dry-run", action="store_true", help="Inspect saved latent shape without loading the VAE."
    )
    decode.set_defaults(func=cmd_decode)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
