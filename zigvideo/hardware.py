from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import platform
import shutil
import subprocess
from typing import Optional

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


@dataclass(frozen=True)
class HardwareProfile:
    os: str
    architecture: str
    ram_gb: float
    gpu_vendor: str
    gpu_name: str
    vram_gb: float
    compute_capability: Optional[float]
    cuda_visible: bool
    supports_bf16: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _ram_gb() -> float:
    if psutil is not None:
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return 0.0


def _query_nvidia_smi() -> tuple[str, float, Optional[float]] | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None

    queries = [
        [exe, "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
        [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    ]

    for cmd in queries:
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            first = completed.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 2:
                name = parts[0]
                vram_gb = round(float(parts[1]) / 1024.0, 2)
                cc = None
                if len(parts) >= 3:
                    try:
                        cc = float(parts[2])
                    except ValueError:
                        cc = None
                return name, vram_gb, cc
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            continue
    return None


def detect_hardware() -> HardwareProfile:
    nvidia = _query_nvidia_smi()
    if nvidia:
        gpu_name, vram_gb, compute_capability = nvidia
        supports_bf16 = compute_capability is not None and compute_capability >= 8.0
        return HardwareProfile(
            os=platform.system(),
            architecture=platform.machine(),
            ram_gb=_ram_gb(),
            gpu_vendor="NVIDIA",
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            compute_capability=compute_capability,
            cuda_visible=True,
            supports_bf16=supports_bf16,
        )

    return HardwareProfile(
        os=platform.system(),
        architecture=platform.machine(),
        ram_gb=_ram_gb(),
        gpu_vendor="CPU/Unknown",
        gpu_name="No NVIDIA GPU detected",
        vram_gb=0.0,
        compute_capability=None,
        cuda_visible=False,
        supports_bf16=False,
    )
