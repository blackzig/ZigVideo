from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from .backends.cogvideox_fp16 import MODEL_REPO, reframe_frames
from .quality import assess_tensor_video, summarize_decoded_tensor


@dataclass(frozen=True)
class DecodeReport:
    input_latents: str
    output: str
    model_repo: str
    vae_dtype: str
    device: str
    offload_strategy: str
    frame_count: int
    fps: int
    aspect: str
    raw_vae: dict
    quality: dict
    load_seconds: float
    decode_seconds: float
    postprocess_seconds: float
    export_seconds: float
    peak_cuda_allocated_gb: float | None
    cuda_physical_vram_gb: float | None
    cuda_peak_exceeds_physical_vram: bool | None
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_latent_file(path: str | Path) -> dict:
    """Inspect saved CogVideoX latent shape and metadata without loading weights."""
    from safetensors import safe_open

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Latents not found: {source}")

    with safe_open(str(source), framework="pt", device="cpu") as file:
        if "latents" not in file.keys():
            raise ValueError("Expected safetensors key 'latents'.")
        shape = list(file.get_slice("latents").get_shape())
        metadata = file.metadata() or {}

    # CogVideoX saves latents [batch, LATENT_FRAMES, CHANNELS, H/8, W/8].
    # In the user's 16-frame test the saved shape is [1, 4, 16, 60, 90].
    if len(shape) != 5 or shape[0] != 1 or shape[2] != 16:
        raise ValueError(
            "Expected CogVideoX latents in shape [1, latent_frames, 16, h, w]. "
            f"Found: {shape}"
        )
    if any(axis < 1 for axis in shape):
        raise ValueError(f"Invalid latent dimensions: {shape}")

    width, height = shape[4] * 8, shape[3] * 8
    for key, measured in (("width", width), ("height", height)):
        if key in metadata and int(metadata[key]) != measured:
            raise ValueError(
                f"Inconsistent {key}: saved metadata={metadata[key]}, "
                f"tensor={measured}"
            )

    return {
        "path": str(source.resolve()),
        "shape": shape,
        "metadata": metadata,
        "native_resolution": f"{width}x{height}",
        "latent_frame_count": shape[1],
    }


def peak_allocation_exceeds_device_memory(
    peak_allocated_gb: float | None, device_total_gb: float | None
) -> bool | None:
    """Flag a CUDA allocator/physical-capacity mismatch, not actual VRAM use."""
    if peak_allocated_gb is None or device_total_gb is None:
        return None
    if peak_allocated_gb < 0 or device_total_gb <= 0:
        raise ValueError("Invalid GPU allocation or memory capacity.")
    return peak_allocated_gb > device_total_gb


def configure_decode_vae(vae, device: str, torch) -> None:
    """Configure safe VAE-only execution without any transformer dependencies."""
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported VAE decode device: {device}")
    vae.enable_tiling()
    vae.enable_slicing()
    if device == "cuda":
        # CogVideoX's decode() bypasses the root forward hook, so
        # block-level offloading of the VAE is unsafe here.
        print("[ZigVideo] Decode-only: VAE leaf-level CPU offload -> CUDA.")
        vae.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=False,
        )


def decode_saved_latents(
    input_latents: str | Path,
    output: str | Path,
    aspect: str = "9:16",
    fps: int = 8,
    cache_dir: str | Path = "models/huggingface",
    device: str = "cpu",
) -> DecodeReport:
    """Run the FP32 VAE only, on CPU or CUDA with leaf-level CPU offload."""
    import torch
    from diffusers import AutoencoderKLCogVideoX
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor
    from safetensors.torch import load_file

    info = inspect_latent_file(input_latents)
    if fps < 1:
        raise ValueError("fps must be positive")
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported VAE decode device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu.")

    output = Path(output)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    load_started = time.perf_counter()
    print(f"[ZigVideo] Decode-only: loading CogVideoX FP32 VAE; device={device}...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        MODEL_REPO,
        subfolder="vae",
        dtype=torch.float32,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    ).eval()
    configure_decode_vae(vae, device, torch)
    load_seconds = time.perf_counter() - load_started

    source = load_file(str(input_latents), device="cpu")["latents"]
    source = source.to(device=device, dtype=torch.float32)
    # Matches Diffusers CogVideoXPipeline.decode_latents, without its transformer.
    z = source.permute(0, 2, 1, 3, 4)
    z = z / float(vae.config.scaling_factor)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    decode_started = time.perf_counter()
    with torch.inference_mode():
        decoded = vae.decode(z).sample
    if device == "cuda":
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started

    postprocess_started = time.perf_counter()
    with torch.inference_mode():
        raw_stats = summarize_decoded_tensor(decoded)
        spatial_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        processor = VideoProcessor(vae_scale_factor=spatial_scale)
        processed = processor.postprocess_video(decoded, output_type="pt")
        frames, quality = assess_tensor_video(processed[0])
    if device == "cuda":
        torch.cuda.synchronize()
    postprocess_seconds = time.perf_counter() - postprocess_started

    peak_cuda_allocated_gb = (
        round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
        if device == "cuda"
        else None
    )
    cuda_physical_vram_gb = (
        round(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3), 3)
        if device == "cuda"
        else None
    )
    cuda_peak_exceeds_physical_vram = peak_allocation_exceeds_device_memory(
        peak_cuda_allocated_gb, cuda_physical_vram_gb
    )
    if cuda_peak_exceeds_physical_vram:
        print(
            "[ZigVideo] WARNING: PyTorch CUDA peak allocation "
            f"({peak_cuda_allocated_gb} GB) exceeds device-reported physical "
            f"capacity ({cuda_physical_vram_gb} GB). Allocator accounting "
            "is not direct physical VRAM residency; GPU/system-memory "
            "oversubscription may be involved. Performance may suffer."
        )

    print(
        "[ZigVideo] Raw VAE: "
        f"nonfinite={raw_stats['nonfinite_fraction']:.3%}, "
        f"sample_mean={raw_stats['sample_mean']}, "
        f"below_minus_one={raw_stats['sample_fraction_below_minus_one']}"
    )
    print(
        "[ZigVideo] Frames: "
        f"quality={quality['status']}, "
        f"max={quality['maximum_pixel_value_0_255']}/255"
    )

    export_started = time.perf_counter()
    frames = reframe_frames(frames, aspect)
    export_to_video(frames, str(output), fps=fps, macro_block_size=8)
    export_seconds = time.perf_counter() - export_started
    print(
        "[ZigVideo] Decode timings: "
        f"load={load_seconds:.2f}s, VAE={decode_seconds:.2f}s, "
        f"postprocess={postprocess_seconds:.2f}s, export={export_seconds:.2f}s, "
        f"peak CUDA allocated={peak_cuda_allocated_gb} GB"
    )
    report = DecodeReport(
        input_latents=info["path"],
        output=str(output.resolve()),
        model_repo=MODEL_REPO,
        vae_dtype="torch.float32",
        device=device,
        offload_strategy="leaf_level" if device == "cuda" else "none",
        frame_count=len(frames),
        fps=fps,
        aspect=aspect,
        raw_vae=raw_stats,
        quality=quality,
        load_seconds=round(load_seconds, 2),
        decode_seconds=round(decode_seconds, 2),
        postprocess_seconds=round(postprocess_seconds, 2),
        export_seconds=round(export_seconds, 2),
        peak_cuda_allocated_gb=peak_cuda_allocated_gb,
        cuda_physical_vram_gb=cuda_physical_vram_gb,
        cuda_peak_exceeds_physical_vram=cuda_peak_exceeds_physical_vram,
        elapsed_seconds=round(time.perf_counter() - started, 2),
    )
    report_path = output.with_suffix(output.suffix + ".json")
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ZigVideo] Decode-only saved: {output.resolve()}")
    print(f"[ZigVideo] Report: {report_path.resolve()}")
    return report
