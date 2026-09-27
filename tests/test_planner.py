from zigvideo.hardware import HardwareProfile
from zigvideo.planner import PerformanceMode, build_plan, classify_hardware


def hw(vram: float, cc: float | None, cuda: bool = True) -> HardwareProfile:
    return HardwareProfile(
        os="Windows",
        architecture="AMD64",
        ram_gb=32,
        gpu_vendor="NVIDIA" if cuda else "CPU/Unknown",
        gpu_name="Test GPU",
        vram_gb=vram,
        compute_capability=cc,
        cuda_visible=cuda,
        supports_bf16=(cc or 0) >= 8.0,
    )


def test_gtx_6gb_uses_legacy_mode():
    machine = hw(6.0, 7.5)
    assert classify_hardware(machine) == PerformanceMode.LEGACY_6GB
    plan = build_plan(machine, task="i2v")
    assert plan.model == "ltxv-2b-distilled"
    assert plan.precision == "fp16"
    assert plan.vae_tiling is True


def test_rtx_6gb_can_choose_framepack():
    machine = hw(6.0, 8.6)
    plan = build_plan(machine, task="i2v")
    assert plan.mode == PerformanceMode.LOW_VRAM
    assert plan.backend == "framepack"


def test_cpu_falls_back_to_hybrid():
    machine = hw(0.0, None, cuda=False)
    plan = build_plan(machine, task="t2v")
    assert plan.backend == "hybrid"
    assert plan.model == "hybrid-keyframes"
