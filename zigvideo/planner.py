from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
import json
from typing import List

from .hardware import HardwareProfile
from .models import MODEL_REGISTRY


class PerformanceMode(str, Enum):
    CPU_FALLBACK = "cpu-fallback"
    ULTRA_LITE = "ultra-lite"
    LEGACY_6GB = "legacy-6gb"
    LOW_VRAM = "low-vram"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass(frozen=True)
class GenerationPlan:
    mode: PerformanceMode
    backend: str
    model: str
    precision: str
    width: int
    height: int
    fps: int
    seconds: float
    frames_per_chunk: int
    offload: str
    vae_tiling: bool
    decode_chunk_size: int
    attention: str
    cache_strategy: str
    estimated_peak_vram_gb: float
    notes: List[str]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mode"] = self.mode.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def classify_hardware(hw: HardwareProfile) -> PerformanceMode:
    if not hw.cuda_visible:
        return PerformanceMode.CPU_FALLBACK
    if hw.vram_gb < 4:
        return PerformanceMode.ULTRA_LITE
    if hw.vram_gb <= 6.5 and (hw.compute_capability or 0) < 8.0:
        return PerformanceMode.LEGACY_6GB
    if hw.vram_gb < 8:
        return PerformanceMode.LOW_VRAM
    if hw.vram_gb < 12:
        return PerformanceMode.BALANCED
    return PerformanceMode.QUALITY


def build_plan(
    hw: HardwareProfile,
    task: str = "i2v",
    seconds: float = 5.0,
    fps: int = 16,
) -> GenerationPlan:
    task = task.lower().strip()
    if task not in {"t2v", "i2v"}:
        raise ValueError("task must be 't2v' or 'i2v'")

    mode = classify_hardware(hw)

    if mode in {PerformanceMode.CPU_FALLBACK, PerformanceMode.ULTRA_LITE}:
        return GenerationPlan(
            mode=mode,
            backend="hybrid",
            model="hybrid-keyframes",
            precision="fp32/cpu or fp16/gpu",
            width=512,
            height=288,
            fps=min(fps, 12),
            seconds=seconds,
            frames_per_chunk=8,
            offload="maximum",
            vae_tiling=True,
            decode_chunk_size=1,
            attention="standard",
            cache_strategy="disk-first",
            estimated_peak_vram_gb=min(max(hw.vram_gb, 0.5), 3.0),
            notes=[
                "Use sparse AI keyframes plus frame interpolation instead of full video diffusion.",
                "Favor short previews; upscale only after motion is accepted.",
            ],
        )

    if mode == PerformanceMode.LEGACY_6GB:
        return GenerationPlan(
            mode=mode,
            backend="diffusers",
            model="cogvideox-2b" if task == "t2v" else "ltxv-2b-distilled",
            precision="fp16",
            width=720 if task == "t2v" else 640,
            height=480 if task == "t2v" else 384,
            fps=min(fps, 8 if task == "t2v" else 16),
            seconds=seconds,
            frames_per_chunk=16,
            offload="sequential-cpu" if task == "t2v" else "group-cpu",
            vae_tiling=True,
            decode_chunk_size=1,
            attention="sdpa",
            cache_strategy="minimal",
            estimated_peak_vram_gb=min(hw.vram_gb * 0.90, 5.4),
            notes=[
                "CogVideoX-2B is the validated T2V path for Turing/GTX-class 6GB hardware because its weights are native FP16.",
                "LTX remains experimental for I2V on pre-BF16 GPUs.",
                "Generate in the model's native geometry, then reframe/upscale for vertical delivery.",
            ],
        )

    if mode == PerformanceMode.LOW_VRAM:
        if hw.supports_bf16 and task in MODEL_REGISTRY["framepack-13b"].tasks:
            return GenerationPlan(
                mode=mode,
                backend="framepack",
                model="framepack-13b",
                precision="bf16/fp16 mixed",
                width=640,
                height=384,
                fps=min(fps, 24),
                seconds=seconds,
                frames_per_chunk=24,
                offload="aggressive",
                vae_tiling=True,
                decode_chunk_size=2,
                attention="sdpa",
                cache_strategy="preview-cache",
                estimated_peak_vram_gb=min(hw.vram_gb * 0.92, 6.0),
                notes=[
                    "Use progressive next-section generation and stream results as soon as frames are ready.",
                    "TeaCache-like acceleration should be preview-only because it can alter results.",
                ],
            )

        return GenerationPlan(
            mode=mode,
            backend="diffusers",
            model="wan21-1.3b" if task == "t2v" else "ltxv-2b-distilled",
            precision="fp16 + quantized weights where compatible",
            width=640,
            height=384,
            fps=min(fps, 16),
            seconds=seconds,
            frames_per_chunk=24,
            offload="group/sequential-cpu",
            vae_tiling=True,
            decode_chunk_size=2,
            attention="sdpa",
            cache_strategy="low-memory",
            estimated_peak_vram_gb=min(hw.vram_gb * 0.92, 7.0),
            notes=["Prefer group offloading for video transformers when supported."],
        )

    if mode == PerformanceMode.BALANCED:
        return GenerationPlan(
            mode=mode,
            backend="diffusers",
            model="wan21-1.3b" if task == "t2v" else "ltxv-2b-distilled",
            precision="fp16/bf16",
            width=832,
            height=480,
            fps=min(fps, 24),
            seconds=seconds,
            frames_per_chunk=32,
            offload="group-cpu",
            vae_tiling=True,
            decode_chunk_size=4,
            attention="sdpa/optimized",
            cache_strategy="balanced",
            estimated_peak_vram_gb=min(hw.vram_gb * 0.90, 10.0),
            notes=["Trade a little host RAM and transfer overhead for stable GPU headroom."],
        )

    return GenerationPlan(
        mode=mode,
        backend="diffusers",
        model="wan21-1.3b" if task == "t2v" else "ltxv-2b-distilled",
        precision="bf16/fp16",
        width=960,
        height=544,
        fps=min(fps, 30),
        seconds=seconds,
        frames_per_chunk=48,
        offload="model-cpu or none",
        vae_tiling=False,
        decode_chunk_size=8,
        attention="optimized",
        cache_strategy="speed",
        estimated_peak_vram_gb=min(hw.vram_gb * 0.90, 14.0),
        notes=["Quality mode can later opt into larger video models and native upscaling."],
    )
