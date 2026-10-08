from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path
import time

from PIL import Image

from ..quality import assess_tensor_video


MODEL_REPO = "THUDM/CogVideoX-2b"
SUPPORTED_ASPECTS = ("16:9", "9:16")


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
) -> dict:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")

    attempt = PRESETS[preset]
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        attempt = replace(attempt, num_inference_steps=num_inference_steps)
    attempt.validate()
    return {
        "backend": "cogvideox-fp16",
        "model_repo": MODEL_REPO,
        "precision": "float16",
        "offload": "sequential CPU offload",
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


def _load_pipeline(cache_dir: Path):
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

    print("[ZigVideo] Enabling sequential CPU offload and VAE tiling...")
    pipeline.enable_sequential_cpu_offload(device="cuda")
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()

    return pipeline


def generate_text_to_video(
    prompt: str,
    output: str | Path,
    preset: str = "ultra-safe",
    aspect: str = "9:16",
    seed: int = 42,
    cache_dir: str | Path = "models/huggingface",
    num_inference_steps: int | None = None,
) -> CogVideoXReport:
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

    attempt = PRESETS[preset]
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        attempt = replace(attempt, num_inference_steps=num_inference_steps)
    attempt.validate()

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    pipeline = _load_pipeline(cache_dir)
    load_seconds = time.perf_counter() - load_started

    _cleanup_cuda(torch)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(
        "[ZigVideo] CogVideoX attempt: "
        f"{attempt.width}x{attempt.height}, {attempt.num_frames} frames, "
        f"{attempt.num_inference_steps} steps @ {attempt.fps} fps"
    )

    inference_started = time.perf_counter()
    with torch.inference_mode():
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
            output_type="pt",
        )
    inference_seconds = time.perf_counter() - inference_started

    # Inspect tensors before NumPy/PIL conversion to catch NaN, Inf,
    # and near-black output that would otherwise be silently exported.
    frames, quality = assess_tensor_video(result.frames[0])
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
