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
    frame_count: int
    fps: int
    aspect: str
    raw_vae: dict
    quality: dict
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


def decode_saved_latents(
    input_latents: str | Path,
    output: str | Path,
    aspect: str = "9:16",
    fps: int = 8,
    cache_dir: str | Path = "models/huggingface",
) -> DecodeReport:
    """Run the VAE only, on CPU in FP32, reusing previously generated latents."""
    import torch
    from diffusers import AutoencoderKLCogVideoX
    from diffusers.utils import export_to_video
    from diffusers.video_processor import VideoProcessor
    from safetensors.torch import load_file

    info = inspect_latent_file(input_latents)
    if fps < 1:
        raise ValueError("fps must be positive")

    output = Path(output)
    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("[ZigVideo] Decode-only: loading CogVideoX VAE on CPU in FP32...")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        MODEL_REPO,
        subfolder="vae",
        dtype=torch.float32,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    ).eval()
    vae.enable_tiling()
    vae.enable_slicing()

    source = load_file(str(input_latents), device="cpu")["latents"]
    source = source.to(dtype=torch.float32)
    # Matches Diffusers CogVideoXPipeline.decode_latents, without its transformer.
    z = source.permute(0, 2, 1, 3, 4)
    z = z / float(vae.config.scaling_factor)

    with torch.inference_mode():
        decoded = vae.decode(z).sample
        raw_stats = summarize_decoded_tensor(decoded)
        spatial_scale = 2 ** (len(vae.config.block_out_channels) - 1)
        processor = VideoProcessor(vae_scale_factor=spatial_scale)
        processed = processor.postprocess_video(decoded, output_type="pt")
        frames, quality = assess_tensor_video(processed[0])

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

    frames = reframe_frames(frames, aspect)
    export_to_video(frames, str(output), fps=fps, macro_block_size=8)
    report = DecodeReport(
        input_latents=info["path"],
        output=str(output.resolve()),
        model_repo=MODEL_REPO,
        vae_dtype="torch.float32",
        device="cpu",
        frame_count=len(frames),
        fps=fps,
        aspect=aspect,
        raw_vae=raw_stats,
        quality=quality,
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
