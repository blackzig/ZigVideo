from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

from PIL import Image

from ..benchmark import summarize_stage_timings
from ..selective_cfg import selective_cfg_transformer, validate_selective_cfg
from ..quality import (
    assess_tensor_video,
    classify_numerical_stage,
    identify_decode_stage,
    summarize_decoded_tensor,
    summarize_latents,
)


MODEL_REPO = "THUDM/CogVideoX-2b"
SUPPORTED_ASPECTS = ("16:9", "9:16")
OFFLOAD_STRATEGIES = ("sequential", "group")


@dataclass(frozen=True)
class CogVideoXAttempt:
    width: int
    height: int
    num_frames: int
    fps: int
    num_inference_steps: int
    guidance_scale: float = 6.0

    def validate(self) -> None:
        if self.width % 8 or self.height % 8:
            raise ValueError("CogVideoX width and height must be divisible by 8.")
        if self.num_frames < 8:
            raise ValueError("CogVideoX preview requires at least 8 frames.")


PRESETS: dict[str, CogVideoXAttempt] = {
    "ultra-safe": CogVideoXAttempt(720, 480, 16, 8, 30, 6.0),
    "safe": CogVideoXAttempt(720, 480, 33, 8, 40, 6.0),
    "balanced": CogVideoXAttempt(720, 480, 49, 8, 50, 6.0),
}


@dataclass(frozen=True)
class CogVideoXLatentReport:
    """Result of denoising without VAE decoding or MP4 export."""

    output: str
    latent_file: str
    model_repo: str
    backend: str
    preset: str
    seed: int
    native_attempt: dict
    offload_strategy: str
    cfg_transformer_batch_factor: int
    cfg_schedule: dict | None
    stage_timings: dict
    latent_checks: list[dict]
    peak_cuda_allocated_gb: float
    load_seconds: float
    inference_seconds: float
    total_seconds: float
    physical_vram_gb: float
    free_vram_after_cleanup_gb: float
    quality: str = "not_evaluated_latents_only"

    def to_dict(self) -> dict:
        return asdict(self)


def transformer_cfg_batch_factor(guidance_scale: float) -> int:
    """Diffusers CogVideoX uses a double batch for guidance_scale > 1."""
    if not 1 <= guidance_scale <= 30:
        raise ValueError("CFG scale must be between 1 and 30.")
    return 2 if guidance_scale > 1.0 else 1


@dataclass(frozen=True)
class CogVideoXReport:
    output: str
    model_repo: str
    backend: str
    precision: str
    preset: str
    aspect: str
    seed: int
    native_attempt: dict
    delivery_resolution: str
    quality: dict
    raw_vae: dict
    vae_precision: str
    offload_strategy: str
    peak_cuda_allocated_gb: float
    stage_timings: dict
    latent_file: str | None
    latent_checks: list[dict]
    load_seconds: float
    inference_seconds: float
    reframe_seconds: float
    export_seconds: float
    total_seconds: float
    physical_vram_gb: float
    free_vram_after_cleanup_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


def generation_preview(
    preset: str,
    cache_dir: str | Path,
    output: str | Path,
    aspect: str = "9:16",
    num_inference_steps: int | None = None,
    vae_fp32: bool = False,
    save_latents: bool = False,
    offload_strategy: str = "sequential",
    cfg_scale: float | None = None,
    latents_only: bool = False,
    cfg_guided_steps: int | None = None,
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")

    if offload_strategy not in OFFLOAD_STRATEGIES:
        raise ValueError(f"Unsupported offload strategy: {offload_strategy}")

    attempt = PRESETS[preset]
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        attempt = replace(attempt, num_inference_steps=num_inference_steps)
    if cfg_scale is not None:
        transformer_cfg_batch_factor(cfg_scale)
        attempt = replace(attempt, guidance_scale=cfg_scale)
    attempt.validate()
    if latents_only and vae_fp32:
        raise ValueError("--vae-fp32 has no effect with --latents-only.")
    validate_selective_cfg(
        cfg_guided_steps, attempt.num_inference_steps, attempt.guidance_scale,
        latents_only, offload_strategy,
    )
    if latents_only:
        save_latents = True
    return {
        "backend": "cogvideox-fp16",
        "model_repo": MODEL_REPO,
        "precision": "float16",
        "vae_precision": "float32" if vae_fp32 else "float16",
        "save_latents": save_latents,
        "latents_only": latents_only,
        "output_mode": "latents_no_vae_or_mp4" if latents_only else "mp4",
        "cfg_transformer_batch_factor": transformer_cfg_batch_factor(attempt.guidance_scale),
        "cfg_guided_steps": cfg_guided_steps,
        "cfg_step_batch_factors": (
            [2] * cfg_guided_steps +
            [1] * (attempt.num_inference_steps - cfg_guided_steps)
            if cfg_guided_steps is not None else None
        ),
        "offload": (
            "sequential CPU offload"
            if offload_strategy == "sequential"
            else "experimental component-wise: transformer block 1/group, VAE/T5 leaf, no streams"
        ),
        "offload_strategy": offload_strategy,
        "vae_tiling": True,
        "native_generation": asdict(attempt),
        "aspect": aspect,
        "delivery_resolution": "360x640" if aspect == "9:16" else "640x360",
        "reframe": (
            "center crop native 720x480 landscape to 9:16, then resize to 360x640"
            if aspect == "9:16"
            else "center crop native 720x480 landscape to 16:9, then resize to 640x360"
        ),
        "cache_dir": str(Path(cache_dir)),
        "output": str(Path(output)),
        "note": (
            "CogVideoX-2B weights are native FP16. The first validation intentionally "
            "keeps generation at the model's 720x480 training regime and reframes only "
            "after generation so quality is not sacrificed for portrait geometry."
        ),
    }


def _cleanup_cuda(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _center_crop_ratio(image: Image.Image, target_width: int, target_height: int) -> Image.Image:
    src_w, src_h = image.size
    target_ratio = target_width / target_height
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        crop_w = max(1, round(src_h * target_ratio))
        left = max(0, (src_w - crop_w) // 2)
        box = (left, 0, left + crop_w, src_h)
    else:
        crop_h = max(1, round(src_w / target_ratio))
        top = max(0, (src_h - crop_h) // 2)
        box = (0, top, src_w, top + crop_h)

    return image.crop(box)


def reframe_frames(frames: list[Image.Image], aspect: str) -> list[Image.Image]:
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")

    if aspect == "9:16":
        target = (360, 640)
    else:
        target = (640, 360)

    reframed: list[Image.Image] = []
    for frame in frames:
        cropped = _center_crop_ratio(frame, *target)
        reframed.append(cropped.resize(target, Image.Resampling.LANCZOS))
    return reframed


def _load_pipeline(
    cache_dir: Path,
    vae_fp32: bool = False,
    offload_strategy: str = "sequential",
):
    import torch
    from diffusers import CogVideoXPipeline

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[ZigVideo] Loading CogVideoX-2B in native FP16...")
    pipeline = CogVideoXPipeline.from_pretrained(
        MODEL_REPO,
        dtype=torch.float16,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    )

    # Important: upcast before installing Accelerate's meta-device hooks.
    # The transformer and text encoder remain in native FP16.
    if vae_fp32:
        print("[ZigVideo] Upcasting VAE weights to FP32 before offload...")
        pipeline.vae.to(dtype=torch.float32)

    configure_offload(pipeline, torch, offload_strategy)
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()

    return pipeline


def configure_offload(pipeline, torch, offload_strategy: str) -> None:
    """Configure each pipeline component without mixing hooks on any one model.

    The full-pipeline block-level strategy left the CogVideoX VAE's conv_in
    weights on CPU while its inputs were on CUDA, because the VAE decode()
    path bypasses some top-level forward hooks. Leaf-level hooks on the VAE
    apply to every convolution even when decode() is called directly.
    """
    if offload_strategy == "sequential":
        print("[ZigVideo] Offload: sequential CPU (established GTX baseline).")
        pipeline.enable_sequential_cpu_offload(device="cuda")
    elif offload_strategy == "group":
        from diffusers.hooks import apply_group_offloading

        onload = torch.device("cuda")
        offload = torch.device("cpu")
        print(
            "[ZigVideo] Offload: experimental component-wise group: "
            "transformer block-level (1/group), VAE + T5 leaf-level, "
            "streams disabled."
        )

        # Do not call pipeline.enable_group_offload with block_level:
        # CogVideoX VAE.decode bypasses the root module's forward hook.
        pipeline.transformer.enable_group_offload(
            onload_device=onload,
            offload_device=offload,
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=False,
        )
        pipeline.vae.enable_group_offload(
            onload_device=onload,
            offload_device=offload,
            offload_type="leaf_level",
            use_stream=False,
        )
        apply_group_offloading(
            pipeline.text_encoder,
            onload_device=onload,
            offload_device=offload,
            offload_type="leaf_level",
            use_stream=False,
        )
    else:
        raise ValueError(f"Unsupported offload strategy: {offload_strategy}")


def save_final_latents(
    latents,
    output: Path,
    attempt: CogVideoXAttempt,
    seed: int,
) -> str:
    """Persist small final latent tensor before VAE decoding can fail."""
    from safetensors.torch import save_file

    destination = output.with_suffix(".latents.safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = latents.detach().to("cpu").contiguous()
    save_file(
        {"latents": data},
        str(destination),
        metadata={
            "model": MODEL_REPO,
            "width": str(attempt.width),
            "height": str(attempt.height),
            "frames": str(attempt.num_frames),
            "steps": str(attempt.num_inference_steps),
            "guidance_scale": str(attempt.guidance_scale),
            "seed": str(seed),
            "note": "Final CogVideoX denoising latents; not decoded video frames.",
        },
    )
    filename = str(destination.resolve())
    print(f"[ZigVideo] Final latents saved before VAE decode: {filename}")
    return filename


def generate_text_to_video(
    prompt: str,
    output: str | Path,
    preset: str = "ultra-safe",
    aspect: str = "9:16",
    seed: int = 42,
    cache_dir: str | Path = "models/huggingface",
    num_inference_steps: int | None = None,
    vae_fp32: bool = False,
    save_latents: bool = False,
    offload_strategy: str = "sequential",
    cfg_scale: float | None = None,
    latents_only: bool = False,
    cfg_guided_steps: int | None = None,
) -> CogVideoXReport | CogVideoXLatentReport:
    import torch
    from diffusers.utils import export_to_video

    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run 'zigvideo torch-check' first.")

    output = Path(output)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)

    if offload_strategy not in OFFLOAD_STRATEGIES:
        raise ValueError(f"Unsupported offload strategy: {offload_strategy}")

    attempt = PRESETS[preset]
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        attempt = replace(attempt, num_inference_steps=num_inference_steps)
    if cfg_scale is not None:
        transformer_cfg_batch_factor(cfg_scale)
        attempt = replace(attempt, guidance_scale=cfg_scale)
    attempt.validate()
    transformer_cfg_batch_factor(attempt.guidance_scale)
    if latents_only and vae_fp32:
        raise ValueError("--vae-fp32 has no effect with --latents-only.")
    validate_selective_cfg(
        cfg_guided_steps, attempt.num_inference_steps, attempt.guidance_scale,
        latents_only, offload_strategy,
    )
    if latents_only:
        save_latents = True

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    pipeline = _load_pipeline(
        cache_dir, vae_fp32=vae_fp32, offload_strategy=offload_strategy
    )
    load_seconds = time.perf_counter() - load_started

    # Inspect VAE output before the diffusers VideoProcessor applies its
    # normalization. Decode into the VAE's own precision (FP16 or FP32).
    raw_vae: dict = {}
    vae_decode_seconds = 0.0

    def monitored_decode_latents(latents):
        nonlocal vae_decode_seconds
        latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents / pipeline.vae_scaling_factor_image
        latents = latents.to(dtype=pipeline.vae.dtype)
        torch.cuda.synchronize()
        vae_started = time.perf_counter()
        video = pipeline.vae.decode(latents).sample
        torch.cuda.synchronize()
        vae_decode_seconds = time.perf_counter() - vae_started
        raw_vae.update(summarize_decoded_tensor(video))
        print(
            "[ZigVideo] Raw VAE output: "
            f"nonfinite={raw_vae['nonfinite_fraction']:.6%}, "
            f"dtype={raw_vae['dtype']}"
        )
        return video

    pipeline.decode_latents = monitored_decode_latents

    _cleanup_cuda(torch)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(
        "[ZigVideo] CogVideoX attempt: "
        f"{attempt.width}x{attempt.height}, {attempt.num_frames} frames, "
        f"{attempt.num_inference_steps} steps @ {attempt.fps} fps; "
        f"CFG={attempt.guidance_scale} "
        f"(transformer batch factor={transformer_cfg_batch_factor(attempt.guidance_scale)}); "
        f"latents-only={latents_only}"
    )

    # Only three scalar diagnostics are collected to avoid significantly
    # slowing down low-VRAM diffusion.
    latent_checks: list[dict] = []
    step_end_elapsed_seconds: list[float] = []
    latent_file: str | None = None
    checkpoints = {0, attempt.num_inference_steps // 2, attempt.num_inference_steps - 1}

    def check_latents(_pipeline, step_index, timestep, callback_kwargs):
        nonlocal latent_file
        # Record a synchronized checkpoint before optional disk writes.
        torch.cuda.synchronize()
        step_end_elapsed_seconds.append(time.perf_counter() - inference_started)
        if save_latents and not latents_only and step_index == attempt.num_inference_steps - 1:
            # Capture on the last denoising step; pipeline() then invokes the
            # VAE internally and could raise before it returns.
            latent_file = save_final_latents(
                callback_kwargs["latents"], output, attempt, seed
            )
        if step_index in checkpoints:
            metrics = summarize_latents(callback_kwargs["latents"], step_index + 1)
            latent_checks.append(metrics)
            print(
                "[ZigVideo] Latents: "
                f"step={step_index + 1}, "
                f"nonfinite={metrics['nonfinite_fraction']:.6%}, "
                f"dtype={metrics['dtype']}"
            )
        return callback_kwargs

    torch.cuda.reset_peak_memory_stats()
    inference_started = time.perf_counter()
    cfg_context = (
        selective_cfg_transformer(
            pipeline.transformer,
            guided_steps=cfg_guided_steps,
            total_steps=attempt.num_inference_steps,
        )
        if cfg_guided_steps is not None else nullcontext(None)
    )
    with torch.inference_mode(), cfg_context as cfg_stats:
        result = pipeline(
            prompt=prompt,
            negative_prompt=(
                "blurry, distorted, deformed, abstract, unrecognizable subject, "
                "low quality, inconsistent motion"
            ),
            width=attempt.width,
            height=attempt.height,
            num_frames=attempt.num_frames,
            num_inference_steps=attempt.num_inference_steps,
            guidance_scale=attempt.guidance_scale,
            generator=generator,
            output_type="latent" if latents_only else "pt",
            callback_on_step_end=check_latents,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    inference_seconds = time.perf_counter() - inference_started
    stage_timings = summarize_stage_timings(
        step_end_elapsed_seconds,
        vae_decode_seconds,
        inference_seconds,
    )
    peak_cuda_allocated_gb = round(
        torch.cuda.max_memory_allocated() / (1024 ** 3), 3
    )
    print(
        "[ZigVideo] Pipeline stages: "
        f"to-latents={stage_timings['time_to_final_latents_seconds']:.2f}s; "
        f"steady-step={stage_timings['subsequent_step_mean_seconds']}s; "
        f"VAE={stage_timings['vae_decode_seconds']:.2f}s; "
        f"pipeline-total={inference_seconds:.2f}s; "
        f"peak CUDA allocated={peak_cuda_allocated_gb:.2f}GB"
    )

    if latents_only:
        # The upstream CogVideoXPipeline returns the native [B, F, C, H, W]
        # tensor for output_type="latent" without invoking VAE or video encoding.
        latent_file = save_final_latents(result.frames, output, attempt, seed)
        del result
        _cleanup_cuda(torch)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        report = CogVideoXLatentReport(
            output=str(Path(latent_file)),
            latent_file=latent_file,
            model_repo=MODEL_REPO,
            backend="cogvideox-fp16-latents-only",
            preset=preset,
            seed=seed,
            native_attempt=asdict(attempt),
            offload_strategy=offload_strategy,
            cfg_transformer_batch_factor=transformer_cfg_batch_factor(attempt.guidance_scale),
            cfg_schedule=cfg_stats.to_dict() if cfg_stats is not None else None,
            stage_timings=stage_timings,
            latent_checks=latent_checks,
            peak_cuda_allocated_gb=peak_cuda_allocated_gb,
            load_seconds=round(load_seconds, 2),
            inference_seconds=round(inference_seconds, 2),
            total_seconds=round(time.perf_counter() - total_started, 2),
            physical_vram_gb=round(total_bytes / (1024 ** 3), 2),
            free_vram_after_cleanup_gb=round(free_bytes / (1024 ** 3), 2),
        )
        report_path = Path(latent_file + ".json")
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"[ZigVideo] Latents-only complete: {latent_file}; "
            f"VAE skipped; MP4 not created; report: {report_path}"
        )
        return report

    # Inspect tensors before NumPy/PIL conversion to catch NaN, Inf,
    # and near-black output that would otherwise be silently exported.
    frames, quality = assess_tensor_video(result.frames[0])
    quality["numerical_stage"] = classify_numerical_stage(latent_checks, quality)
    quality["decode_stage"] = identify_decode_stage(latent_checks, raw_vae, quality)
    print(
        "[ZigVideo] Quality preflight: "
        f"status={quality['status']}, "
        f"nonfinite={quality['nonfinite_fraction']:.6%}, "
        f"bright_pixels={quality['bright_pixel_fraction']:.2%}"
    )
    if quality["status"] != "plausible":
        print(
            "[ZigVideo] WARNING: Video may be unusable. Check the "
            "quality diagnostics in the JSON report before delivery."
        )

    reframe_started = time.perf_counter()
    frames = reframe_frames(frames, aspect)
    reframe_seconds = time.perf_counter() - reframe_started

    export_started = time.perf_counter()
    export_to_video(
        frames,
        str(output),
        fps=attempt.fps,
        macro_block_size=8,
    )
    export_seconds = time.perf_counter() - export_started

    del result
    _cleanup_cuda(torch)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_seconds = time.perf_counter() - total_started

    report = CogVideoXReport(
        output=str(output.resolve()),
        model_repo=MODEL_REPO,
        backend="cogvideox-fp16",
        precision="float16",
        preset=preset,
        aspect=aspect,
        seed=seed,
        native_attempt=asdict(attempt),
        delivery_resolution=f"{frames[0].width}x{frames[0].height}",
        quality=quality,
        raw_vae=raw_vae,
        vae_precision="float32" if vae_fp32 else "float16",
        offload_strategy=offload_strategy,
        peak_cuda_allocated_gb=peak_cuda_allocated_gb,
        stage_timings=stage_timings,
        latent_file=latent_file,
        latent_checks=latent_checks,
        load_seconds=round(load_seconds, 2),
        inference_seconds=round(inference_seconds, 2),
        reframe_seconds=round(reframe_seconds, 2),
        export_seconds=round(export_seconds, 2),
        total_seconds=round(total_seconds, 2),
        physical_vram_gb=round(total_bytes / (1024**3), 2),
        free_vram_after_cleanup_gb=round(free_bytes / (1024**3), 2),
    )

    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ZigVideo] Saved: {output.resolve()}")
    print(f"[ZigVideo] Report: {report_path.resolve()}")
    return report
