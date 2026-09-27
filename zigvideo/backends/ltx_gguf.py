from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import time
from typing import Iterable


PIPELINE_REPO = "Lightricks/LTX-Video"
GGUF_REPO = "city96/LTX-Video-0.9.6-distilled-gguf"
GGUF_FILENAME = "ltxv-2b-0.9.6-distilled-04-25-Q3_K_S.gguf"


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


PRESETS: dict[str, GenerationAttempt] = {
    "ultra-safe": GenerationAttempt(320, 192, 9, 8, 8, 64),
    "safe": GenerationAttempt(384, 256, 17, 8, 8, 96),
    "balanced": GenerationAttempt(512, 320, 17, 12, 8, 128),
}

EMERGENCY_ATTEMPTS: tuple[GenerationAttempt, ...] = (
    GenerationAttempt(256, 160, 9, 8, 8, 64),
    GenerationAttempt(256, 128, 9, 8, 8, 48),
    GenerationAttempt(192, 128, 9, 8, 8, 48),
)


@dataclass(frozen=True)
class GenerationReport:
    output: str
    model_repo: str
    model_file: str
    preset: str
    seed: int
    successful_attempt: dict
    elapsed_seconds: float
    peak_vram_gb: float
    free_vram_after_gb: float
    retries: int

    def to_dict(self) -> dict:
        return asdict(self)


def build_attempt_ladder(preset: str) -> list[GenerationAttempt]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    ordered: list[GenerationAttempt] = [PRESETS[preset], *EMERGENCY_ATTEMPTS]
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


def generation_preview(preset: str, cache_dir: str | Path, output: str | Path) -> dict:
    attempts = build_attempt_ladder(preset)
    return {
        "backend": "ltx-gguf",
        "pipeline_repo": PIPELINE_REPO,
        "transformer_repo": GGUF_REPO,
        "transformer_file": GGUF_FILENAME,
        "compute_dtype": "float16",
        "quantization": "GGUF Q3_K_S",
        "offload": "sequential CPU offload",
        "vae_tiling": True,
        "cache_dir": str(Path(cache_dir)),
        "output": str(Path(output)),
        "attempts": [asdict(a) for a in attempts],
        "note": (
            "First run downloads several GB of model components. "
            "Models are cached and are not committed to Git."
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


def _load_pipeline(cache_dir: Path):
    import torch
    from diffusers import AutoModel, GGUFQuantizationConfig, LTXPipeline
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ZigVideo] Model cache: {cache_dir.resolve()}")
    print("[ZigVideo] Downloading/checking the 2B distilled GGUF transformer...")
    gguf_path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILENAME,
        cache_dir=str(cache_dir),
    )

    print("[ZigVideo] Loading quantized transformer on CPU (FP16 compute)...")
    transformer = AutoModel.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.float16),
        config=PIPELINE_REPO,
        subfolder="transformer",
        dtype=torch.float16,
    )

    print("[ZigVideo] Loading LTX pipeline components...")
    pipeline = LTXPipeline.from_pretrained(
        PIPELINE_REPO,
        transformer=transformer,
        dtype=torch.float16,
        cache_dir=str(cache_dir),
        low_cpu_mem_usage=True,
    )

    # A 6 GB Turing card cannot safely hold the full text encoder or all pipeline
    # components at once. Leaf-level sequential offload minimizes accelerator
    # memory at the cost of speed.
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
    seed: int = 42,
    cache_dir: str | Path = "models/huggingface",
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

    attempts = build_attempt_ladder(preset)
    pipeline = _load_pipeline(cache_dir)

    started = time.perf_counter()
    last_oom: BaseException | None = None

    for index, attempt in enumerate(attempts):
        print(
            "[ZigVideo] Attempt "
            f"{index + 1}/{len(attempts)}: "
            f"{attempt.width}x{attempt.height}, {attempt.num_frames} frames, "
            f"{attempt.num_inference_steps} steps"
        )

        _cleanup_cuda(torch)
        torch.cuda.reset_peak_memory_stats()

        try:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = pipeline(
                prompt=prompt,
                negative_prompt=None,
                width=attempt.width,
                height=attempt.height,
                num_frames=attempt.num_frames,
                num_inference_steps=attempt.num_inference_steps,
                guidance_scale=1.0,
                decode_timestep=0.05,
                decode_noise_scale=0.025,
                max_sequence_length=attempt.max_sequence_length,
                generator=generator,
                output_type="pil",
            )

            frames = result.frames[0]
            export_to_video(frames, str(output), fps=attempt.fps)

            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            free_bytes, _ = torch.cuda.mem_get_info()
            free_vram = free_bytes / (1024**3)
            elapsed = time.perf_counter() - started

            report = GenerationReport(
                output=str(output.resolve()),
                model_repo=GGUF_REPO,
                model_file=GGUF_FILENAME,
                preset=preset,
                seed=seed,
                successful_attempt=asdict(attempt),
                elapsed_seconds=round(elapsed, 2),
                peak_vram_gb=round(peak_vram, 2),
                free_vram_after_gb=round(free_vram, 2),
                retries=index,
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

        print("[ZigVideo] CUDA OOM detected. Releasing VRAM and retrying smaller settings...")
        _cleanup_cuda(torch)

    raise RuntimeError(
        "All low-VRAM attempts failed with CUDA OOM. "
        "Close GPU-using applications and retry. "
        f"Last error: {last_oom}"
    )
