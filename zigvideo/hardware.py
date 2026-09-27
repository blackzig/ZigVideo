from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
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
    python_version: str = platform.python_version()
    python_bits: int = struct.calcsize("P") * 8
    nvidia_smi_path: Optional[str] = None
    ai_runtime_ready: bool = True
    environment_warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def _ram_gb() -> float:
    if psutil is not None:
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    return 0.0


def _environment_status() -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    python_bits = struct.calcsize("P") * 8

    if python_bits != 64:
        warnings.append(
            "Python is 32-bit. ZigVideo AI backends require a 64-bit Python runtime."
        )

    if platform.system() == "Windows":
        major, minor = sys.version_info[:2]
        if (major, minor) < (3, 9) or (major, minor) > (3, 12):
            warnings.append(
                "Current PyTorch Windows binaries support Python 3.9-3.12; "
                "ZigVideo recommends Python 3.12 x64 for inference."
            )

    return len(warnings) == 0, tuple(warnings)


def _candidate_nvidia_smi_paths() -> list[str]:
    candidates: list[str] = []

    from_path = shutil.which("nvidia-smi")
    if from_path:
        candidates.append(from_path)

    if platform.system() == "Windows":
        windir = Path(os.environ.get("WINDIR", r"C:\Windows"))

        # Sysnative lets a 32-bit Python process access the real 64-bit
        # System32 directory instead of being redirected to SysWOW64.
        candidates.extend(
            [
                str(windir / "Sysnative" / "nvidia-smi.exe"),
                str(windir / "System32" / "nvidia-smi.exe"),
                str(
                    Path(
                        os.environ.get(
                            "ProgramFiles",
                            r"C:\Program Files",
                        )
                    )
                    / "NVIDIA Corporation"
                    / "NVSMI"
                    / "nvidia-smi.exe"
                ),
            ]
        )

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _query_nvidia_smi() -> tuple[str, float, Optional[float], str] | None:
    for exe in _candidate_nvidia_smi_paths():
        if not os.path.isfile(exe):
            continue

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
                lines = completed.stdout.strip().splitlines()
                if not lines:
                    continue
                parts = [p.strip() for p in lines[0].split(",")]
                if len(parts) >= 2:
                    name = parts[0]
                    vram_gb = round(float(parts[1]) / 1024.0, 2)
                    cc = None
                    if len(parts) >= 3:
                        try:
                            cc = float(parts[2])
                        except ValueError:
                            cc = None
                    return name, vram_gb, cc, exe
            except (subprocess.SubprocessError, OSError, ValueError, IndexError):
                continue
    return None


def detect_hardware() -> HardwareProfile:
    python_version = platform.python_version()
    python_bits = struct.calcsize("P") * 8
    ai_runtime_ready, environment_warnings = _environment_status()

    nvidia = _query_nvidia_smi()
    if nvidia:
        gpu_name, vram_gb, compute_capability, smi_path = nvidia
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
            python_version=python_version,
            python_bits=python_bits,
            nvidia_smi_path=smi_path,
            ai_runtime_ready=ai_runtime_ready,
            environment_warnings=environment_warnings,
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
        python_version=python_version,
        python_bits=python_bits,
        nvidia_smi_path=None,
        ai_runtime_ready=ai_runtime_ready,
        environment_warnings=environment_warnings,
    )
