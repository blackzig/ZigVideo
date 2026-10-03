from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    backend: str
    tasks: FrozenSet[str]
    upstream_vram_gb: float | None
    project_target_vram_gb: float
    requires_bf16: bool = False
    min_compute_capability: float | None = None
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "hybrid-keyframes": ModelSpec(
        key="hybrid-keyframes",
        display_name="Hybrid Keyframes + Interpolation",
        backend="hybrid",
        tasks=frozenset({"t2v", "i2v"}),
        upstream_vram_gb=None,
        project_target_vram_gb=2.0,
        notes="Fallback path for very weak GPUs/CPU: generate sparse AI keyframes and interpolate.",
    ),
    "cogvideox-2b": ModelSpec(
        key="cogvideox-2b",
        display_name="CogVideoX 2B",
        backend="diffusers",
        tasks=frozenset({"t2v"}),
        upstream_vram_gb=4.0,
        project_target_vram_gb=4.0,
        requires_bf16=False,
        notes=(
            "FP16-native legacy GPU candidate. Diffusers documents sequential CPU "
            "offload below 4GB VRAM; validate quality before promoting to default."
        ),
    ),
    "ltxv-2b-distilled": ModelSpec(
        key="ltxv-2b-distilled",
        display_name="LTX-Video 2B Distilled",
        backend="diffusers",
        tasks=frozenset({"t2v", "i2v"}),
        upstream_vram_gb=None,
        project_target_vram_gb=4.5,
        requires_bf16=False,
        notes="Primary legacy/low-VRAM research target using FP16, chunking, tiling and CPU offload.",
    ),
    "framepack-13b": ModelSpec(
        key="framepack-13b",
        display_name="FramePack 13B",
        backend="framepack",
        tasks=frozenset({"i2v", "t2v"}),
        upstream_vram_gb=6.0,
        project_target_vram_gb=6.0,
        requires_bf16=True,
        min_compute_capability=8.0,
        notes="Upstream targets RTX 30/40/50-series and 6GB+ VRAM.",
    ),
    "wan21-1.3b": ModelSpec(
        key="wan21-1.3b",
        display_name="Wan 2.1 T2V 1.3B",
        backend="diffusers",
        tasks=frozenset({"t2v"}),
        upstream_vram_gb=8.19,
        project_target_vram_gb=6.0,
        requires_bf16=False,
        notes="Official baseline is 8.19GB; ZigVideo experiments with quantization/offload below that.",
    ),
}
