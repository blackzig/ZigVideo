from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Optional


@dataclass(frozen=True)
class TorchRuntimeReport:
    torch_installed: bool
    torch_version: Optional[str]
    torch_cuda_build: Optional[str]
    cuda_available: bool
    device_name: Optional[str]
    compute_capability: Optional[str]
    total_vram_gb: Optional[float]
    free_vram_gb: Optional[float]
    fp16_smoke_test: bool
    error: Optional[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def inspect_torch_runtime(run_smoke_test: bool = True) -> TorchRuntimeReport:
    try:
        import torch
    except Exception as exc:
        return TorchRuntimeReport(
            torch_installed=False,
            torch_version=None,
            torch_cuda_build=None,
            cuda_available=False,
            device_name=None,
            compute_capability=None,
            total_vram_gb=None,
            free_vram_gb=None,
            fp16_smoke_test=False,
            error=f"PyTorch is not installed or failed to import: {exc}",
        )

    if not torch.cuda.is_available():
        return TorchRuntimeReport(
            torch_installed=True,
            torch_version=torch.__version__,
            torch_cuda_build=getattr(torch.version, "cuda", None),
            cuda_available=False,
            device_name=None,
            compute_capability=None,
            total_vram_gb=None,
            free_vram_gb=None,
            fp16_smoke_test=False,
            error="PyTorch imported, but CUDA is not available.",
        )

    try:
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        smoke_ok = False

        if run_smoke_test:
            # Small FP16 CUDA operation: enough to validate that kernels really execute
            # on the GPU without consuming meaningful VRAM.
            with torch.inference_mode():
                a = torch.randn((512, 512), device=device, dtype=torch.float16)
                b = torch.randn((512, 512), device=device, dtype=torch.float16)
                c = a @ b
                torch.cuda.synchronize(device)
                smoke_ok = bool(torch.isfinite(c).all().item())
                del a, b, c
            torch.cuda.empty_cache()
        else:
            smoke_ok = True

        return TorchRuntimeReport(
            torch_installed=True,
            torch_version=torch.__version__,
            torch_cuda_build=getattr(torch.version, "cuda", None),
            cuda_available=True,
            device_name=name,
            compute_capability=f"{major}.{minor}",
            total_vram_gb=round(total_bytes / (1024 ** 3), 2),
            free_vram_gb=round(free_bytes / (1024 ** 3), 2),
            fp16_smoke_test=smoke_ok,
            error=None,
        )
    except Exception as exc:
        return TorchRuntimeReport(
            torch_installed=True,
            torch_version=torch.__version__,
            torch_cuda_build=getattr(torch.version, "cuda", None),
            cuda_available=True,
            device_name=None,
            compute_capability=None,
            total_vram_gb=None,
            free_vram_gb=None,
            fp16_smoke_test=False,
            error=f"CUDA runtime test failed: {type(exc).__name__}: {exc}",
        )
