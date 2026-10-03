from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path
import time


PIPELINE_REPO = "Lightricks/LTX-Video"
GGUF_REPO = "city96/LTX-Video-0.9.6-distilled-gguf"
GGUF_FILENAME = "ltxv-2b-0.9.6-distilled-04-25-Q5_K_M.gguf"
QUANTIZATION_LABEL = "GGUF Q5_K_M"
SUPPORTED_ASPECTS = ("16:9", "9:16")


@dataclass(frozen=True)
class GenerationAttempt:
    width: int
    height: int
    num_frames: int
    fps: int
    num_inference_steps: int
    max_sequence_length: int = 64

    def validate(self) -> None:
        if self.width % 32 or self.height % 32:
            raise ValueError("LTX width and height must be divisible by 32.")
        if (self.num_frames - 1) % 8:
            raise ValueError("LTX frame count must be N*8+1 (9, 17, 25, ...).")


# LTX 0.9.x is trained around a 512-resolution bucket. For ~16:9 that bucket
# is 640x384. The old 320x192 compatibility probe ran but produced unusable
# smear, so low-VRAM mode now saves memory with temporal length/offload rather
# than halving the spatial canvas below the model's useful operating range.
PRESETS: dict[str, GenerationAttempt] = {
    "ultra-safe": GenerationAttempt(640, 384, 9, 8, 8, 64),
    "safe": GenerationAttempt(640, 384, 17, 8, 8, 96),
    "balanced": GenerationAttempt(704, 480, 17, 12, 8, 128),
}

EMERGENCY_ATTEMPTS: tuple[GenerationAttempt, ...] = (
    GenerationAttempt(576, 352, 9, 8, 8, 64),
    GenerationAttempt(512, 320, 9, 8, 8, 64),
)


@dataclass(frozen=True)
class GenerationReport:
    output: str
    model_repo: str
    model_file: str
    quantization: str
    preset: str
    aspect: str
    seed: int
    successful_attempt: dict
    load_seconds: float
    denoise_seconds: float
    decode_seconds: float
    export_seconds: float
    total_seconds: float
    peak_torch_allocated_gb: float
    physical_vram_gb: float
    free_vram_after_cleanup_gb: float
    retries: int
    precision_note: str

    def to_dict(self) -> dict:
        return asdict(self)


def _orient_attempt(attempt: GenerationAttempt, aspect: str) -> GenerationAttempt:
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")
    if aspect == "16:9":
        return attempt
    return GenerationAttempt(
        width=attempt.height,
        height=attempt.width,
        num_frames=attempt.num_frames,
        fps=attempt.fps,
        num_inference_steps=attempt.num_inference_steps,
        max_sequence_length=attempt.max_sequence_length,
    )


def build_attempt_ladder(
    preset: str,
    aspect: str = "16:9",
) -> list[GenerationAttempt]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    if aspect not in SUPPORTED_ASPECTS:
        raise ValueError(f"Unsupported aspect ratio: {aspect}")

    ordered: list[GenerationAttempt] = [
        _orient_attempt(attempt, aspect)
        for attempt in [PRESETS[preset], *EMERGENCY_ATTEMPTS]
    ]
    unique: list[GenerationAttempt] = []
    seen: set[tuple[int, int, int, int, int, int]] = set()

    for attempt in ordered:
        attempt.validate()
        key = (
            attempt.width,
            attempt.height,
            attempt.num_frames,
            attempt.fps,
            attempt.num_inference_steps,
            attempt.max_sequence_length,
        )
        if key not in seen:
            seen.add(key)
            unique.append(attempt)
    return unique


def generation_preview(
    preset: str,
    cache_dir: str | Path,
    output: str | Path,
    aspect: str = "16:9",
    num_inference_steps: int | None = None,
) -> dict:
    attempts = build_attempt_ladder(preset, aspect)
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        attempts = [
            replace(attempt, num_inference_steps=num_inference_steps)
            for attempt in attempts
        ]
    return {
        "backend": "ltx-gguf",
        "aspect": aspect,
        "recommended_delivery": (
            "720x1280 or 1080x1920 after upscale"
            if aspect == "9:16"
            else "1280x720 or 1920x1080 after upscale"
        ),
        "pipeline_repo": PIPELINE_REPO,
        "transformer_repo": GGUF_REPO,
        "transformer_file": GGUF_FILENAME,
        "transformer_loader": "LTXVideoTransformer3DModel.from_single_file",
        "compute_dtype": "float16",
        "quantization": QUANTIZATION_LABEL,
        "scheduler": "LTX FlowMatch + stochastic sampling",
        "offload": "GGUF-aware group offload; staged whole-VAE decode",
        "vae_dtype": "float32",
        "vae_tiling": True,
        "cache_dir": str(Path(cache_dir)),
        "output": str(Path(output)),
        "attempts": [asdict(a) for a in attempts],
        "note": (
            "The 6 GB legacy profile starts at the model's 640x384 low-resolution "
            "bucket for 16:9 or the native vertical 384x640 bucket for 9:16. "
            "Q5_K_M replaces the earlier Q3_K_S quality probe. The transformer "
            "still computes in FP16 on pre-BF16 GPUs, while the VAE is kept in FP32 "
            "to avoid precision loss during decode."
        ),
    }


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "gpu" in text)


def _cleanup_cuda(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _load_quantized_transformer(gguf_path: str):
    import torch
    from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel

    loader = getattr(LTXVideoTransformer3DModel, "from_single_file", None)
    if loader is None:
        raise RuntimeError(
            "Installed Diffusers does not expose "
            "LTXVideoTransformer3DModel.from_single_file. "
            "Run 'zigvideo ai-check' after updating dependencies."
        )

    return loader(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        config=PIPELINE_REPO,
        subfolder="transformer",
        dtype=torch.float16,
    )


def _load_pipeline(cache_dir: Path):
    import torch
    from diffusers import AutoencoderKLLTXVideo, FlowMatchEulerDiscreteScheduler, LTXPipeline
    from diffusers.hooks import apply_group_offloading
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ZigVideo] Model cache: {cache_dir.resolve()}")
    print(f"[ZigVideo] Downloading/checking {QUANTIZATION_LABEL} distilled transformer...")
    gguf_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILENAME,
        cache_dir=str(cache_dir),
    )

    print("[ZigVideo] Loading quantized LTX transformer on CPU (FP16 compute)...")
    transformer = _load_quantized_transformer(gguf_path)

    print("[ZigVideo] Loading LTX VAE in FP32 for numerically stable decode...")
    vae = AutoencoderKLLTXVideo.from_pretrained(
        PIPELINE_REPO,
        subfolder="vae",
        dtype=torch.float32,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    )

    print("[ZigVideo] Loading remaining LTX pipeline components...")
    pipeline = LTXPipeline.from_pretrained(
        PIPELINE_REPO,
        transformer=transformer,
        vae=vae,
        dtype=torch.float16,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    )

    if pipeline.vae.dtype != torch.float32:
        pipeline.vae.to(dtype=torch.float32)

    # LTX 0.9.6 distilled's reference config uses stochastic sampling. The
    # Hugging Face base pipeline scheduler has the correct 0.95/2.05 dynamic
    # shift and 0.1 terminal shift, but stochastic_sampling defaults to False.
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        stochastic_sampling=True,
    )

    # Sequential CPU offload is unsafe for GGUFParameter because it recreates
    # parameters on meta and can lose quant_type metadata. Group offloading
    # moves the existing quantized parameters and preserves GGUF metadata.
    print("[ZigVideo] Enabling GGUF-aware group offload...")
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")

    pipeline.transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=False,
    )

    apply_group_offloading(
        pipeline.text_encoder,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=1,
        use_stream=False,
    )

    # Do not leaf-offload the VAE during denoising. It stays on CPU and is
    # moved as a whole to CUDA only after the transformer is finished. The
    # previous leaf-level VAE path was extremely slow on PCIe/legacy hardware.
    if hasattr(pipeline.vae, "enable_tiling"):
        pipeline.vae.enable_tiling()
    if hasattr(pipeline.vae, "enable_slicing"):
        pipeline.vae.enable_slicing()

    return pipeline


def _decode_latents_staged(pipeline, packed_latents, attempt, generator):
    import torch
    from diffusers.hooks import apply_group_offloading

    latent_num_frames = (
        (attempt.num_frames - 1) // pipeline.vae_temporal_compression_ratio + 1
    )
    latent_height = attempt.height // pipeline.vae_spatial_compression_ratio
    latent_width = attempt.width // pipeline.vae_spatial_compression_ratio

    # Unpack and denormalize on CPU so denoising allocations can be released
    # before the VAE occupies the GPU.
    latents = pipeline._unpack_latents(
        packed_latents.to("cpu"),
        latent_num_frames,
        latent_height,
        latent_width,
        pipeline.transformer_spatial_patch_size,
        pipeline.transformer_temporal_patch_size,
    )
    latents = pipeline._denormalize_latents(
        latents,
        pipeline.vae.latents_mean,
        pipeline.vae.latents_std,
        pipeline.vae.config.scaling_factor,
    )

    _cleanup_cuda(torch)

    def prepare_decode_tensors():
        target_dtype = pipeline.vae.dtype
        latents_gpu = latents.to(device="cuda", dtype=target_dtype)
        noise_cpu = torch.randn(
            latents.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        noise = noise_cpu.to(device="cuda", dtype=target_dtype)
        latents_gpu = (1.0 - 0.025) * latents_gpu + 0.025 * noise
        timestep = torch.tensor([0.05], device="cuda", dtype=target_dtype)
        return latents_gpu, timestep

    try:
        print("[ZigVideo] Moving the tiled VAE to CUDA for staged decode...")
        pipeline.vae.to("cuda")
        latents_gpu, timestep = prepare_decode_tensors()
        with torch.inference_mode():
            video = pipeline.vae.decode(
                latents_gpu,
                timestep,
                return_dict=False,
            )[0]
    except torch.OutOfMemoryError:
        print(
            "[ZigVideo] Whole-VAE decode did not fit. Falling back to leaf-level "
            "VAE group offload..."
        )
        try:
            pipeline.vae.to("cpu")
        except Exception:
            pass
        _cleanup_cuda(torch)

        apply_group_offloading(
            pipeline.vae,
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )
        latents_gpu, timestep = prepare_decode_tensors()
        with torch.inference_mode():
            video = pipeline.vae.decode(
                latents_gpu,
                timestep,
                return_dict=False,
            )[0]

    video = video.detach()
    batch_frames = pipeline.video_processor.postprocess_video(
        video,
        output_type="pil",
    )
    if not batch_frames or not isinstance(batch_frames[0], list):
        raise RuntimeError(
            "Unexpected Diffusers video postprocess output; expected a batch of frame lists."
        )
    frames = batch_frames[0]

    try:
        pipeline.vae.to("cpu")
    except Exception:
        pass
    del video
    _cleanup_cuda(torch)
    return frames


def generate_text_to_video(
    prompt: str,
    output: str | Path,
    preset: str = "ultra-safe",
    aspect: str = "16:9",
    seed: int = 42,
    cache_dir: str | Path = "models/huggingface",
    num_inference_steps: int | None = None,
) -> GenerationReport:
    import torch
    from diffusers.utils import export_to_video

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run 'zigvideo torch-check' first.")

    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) < (7, 0):
        raise RuntimeError(
            f"Experimental LTX-GGUF backend requires CUDA capability >= 7.0; found {major}.{minor}."
        )

    output = Path(output)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)

    total_started = time.perf_counter()
    load_started = time.perf_counter()
    attempts = build_attempt_ladder(preset, aspect)
    if num_inference_steps is not None:
        if num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")
        attempts = [
            replace(attempt, num_inference_steps=num_inference_steps)
            for attempt in attempts
        ]
    pipeline = _load_pipeline(cache_dir)
    load_seconds = time.perf_counter() - load_started

    last_oom: BaseException | None = None

    for index, attempt in enumerate(attempts):
        print(
            "[ZigVideo] Attempt "
            f"{index + 1}/{len(attempts)}: "
            f"{attempt.width}x{attempt.height}, {attempt.num_frames} frames, "
            f"{attempt.num_inference_steps} steps @ {attempt.fps} fps"
        )

        _cleanup_cuda(torch)
        torch.cuda.reset_peak_memory_stats()

        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)

            denoise_started = time.perf_counter()
            result = pipeline(
                prompt=prompt,
                negative_prompt=None,
                width=attempt.width,
                height=attempt.height,
                num_frames=attempt.num_frames,
                frame_rate=attempt.fps,
                num_inference_steps=attempt.num_inference_steps,
                guidance_scale=1.0,
                max_sequence_length=attempt.max_sequence_length,
                generator=generator,
                output_type="latent",
            )
            denoise_seconds = time.perf_counter() - denoise_started

            packed_latents = result.frames.detach().to("cpu")
            del result
            _cleanup_cuda(torch)

            decode_started = time.perf_counter()
            frames = _decode_latents_staged(
                pipeline,
                packed_latents,
                attempt,
                generator,
            )
            decode_seconds = time.perf_counter() - decode_started

            export_started = time.perf_counter()
            export_to_video(frames, str(output), fps=attempt.fps)
            export_seconds = time.perf_counter() - export_started

            _cleanup_cuda(torch)
            peak_torch = torch.cuda.max_memory_allocated() / (1024**3)
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_vram = free_bytes / (1024**3)
            physical_vram = total_bytes / (1024**3)
            total_seconds = time.perf_counter() - total_started

            precision_note = (
                "GTX/Turing lacks native BF16; LTX's transformer is running in FP16. "
                "This is a legacy compatibility path and remains quality-experimental."
                if major < 8
                else "BF16-capable hardware detected, but this experimental GGUF path currently uses FP16 compute."
            )

            report = GenerationReport(
                output=str(output.resolve()),
                model_repo=GGUF_REPO,
                model_file=GGUF_FILENAME,
                quantization=QUANTIZATION_LABEL,
                preset=preset,
                aspect=aspect,
                seed=seed,
                successful_attempt=asdict(attempt),
                load_seconds=round(load_seconds, 2),
                denoise_seconds=round(denoise_seconds, 2),
                decode_seconds=round(decode_seconds, 2),
                export_seconds=round(export_seconds, 2),
                total_seconds=round(total_seconds, 2),
                peak_torch_allocated_gb=round(peak_torch, 2),
                physical_vram_gb=round(physical_vram, 2),
                free_vram_after_cleanup_gb=round(free_vram, 2),
                retries=index,
                precision_note=precision_note,
            )

            report_path = output.with_suffix(output.suffix + ".json")
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"[ZigVideo] Saved: {output.resolve()}")
            print(f"[ZigVideo] Report: {report_path.resolve()}")
            return report

        except torch.OutOfMemoryError as exc:
            last_oom = exc
        except RuntimeError as exc:
            if not _is_cuda_oom(exc):
                raise
            last_oom = exc

        print(
            "[ZigVideo] CUDA OOM detected. Releasing VRAM and retrying the "
            "next quality-preserving fallback..."
        )
        _cleanup_cuda(torch)

    raise RuntimeError(
        "All low-VRAM attempts failed with CUDA OOM. "
        "Close GPU-using applications and retry. "
        f"Last error: {last_oom}"
    )
