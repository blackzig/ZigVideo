from PIL import Image

from zigvideo.backends.cogvideox_fp16 import (
    MODEL_REPO,
    PRESETS,
    generation_preview,
    reframe_frames,
)


def test_cogvideox_ultra_safe_matches_native_training_resolution():
    attempt = PRESETS["ultra-safe"]
    attempt.validate()
    assert attempt.width == 720
    assert attempt.height == 480
    assert attempt.num_frames == 16
    assert attempt.fps == 8
    assert attempt.num_inference_steps == 30


def test_cogvideox_preview_is_fp16_and_low_vram(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "cog.mp4",
        aspect="9:16",
    )
    assert preview["backend"] == "cogvideox-fp16"
    assert preview["model_repo"] == MODEL_REPO
    assert preview["precision"] == "float16"
    assert preview["offload"] == "sequential CPU offload"
    assert preview["native_generation"]["width"] == 720
    assert preview["native_generation"]["height"] == 480


def test_cogvideox_short_reframe_is_exact_9_16():
    source = Image.new("RGB", (720, 480))
    frames = reframe_frames([source], "9:16")
    assert len(frames) == 1
    assert frames[0].size == (360, 640)
    assert frames[0].width * 16 == frames[0].height * 9


def test_installed_diffusers_exposes_cogvideox_pipeline():
    from diffusers import CogVideoXPipeline

    assert CogVideoXPipeline is not None


def test_cogvideox_preview_allows_step_override(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "preview.mp4",
        aspect="9:16",
        num_inference_steps=12,
    )
    assert preview["native_generation"]["num_inference_steps"] == 12
    assert preview["native_generation"]["num_frames"] == 16
    assert preview["native_generation"]["width"] == 720
    assert preview["native_generation"]["height"] == 480


def test_quality_preflight_detects_black_output():
    import numpy as np
    from zigvideo.quality import assess_tensor_video

    source = np.zeros((2, 3, 8, 8), dtype=np.float32)
    frames, metrics = assess_tensor_video(source)
    assert len(frames) == 2
    assert metrics["status"] == "near_black"
    assert metrics["maximum_pixel_value_0_255"] == 0


def test_quality_preflight_reports_nan_inf():
    import numpy as np
    from zigvideo.quality import assess_tensor_video

    source = np.ones((2, 3, 8, 8), dtype=np.float32)
    source[0, 0, 0, 0] = np.nan
    source[1, 1, 0, 0] = np.inf
    frames, metrics = assess_tensor_video(source)
    assert metrics["status"] == "nonfinite_detected"
    assert metrics["nonfinite_fraction"] > 0
    assert frames[0].getpixel((0, 0))[0] == 0
    assert frames[1].getpixel((0, 0))[1] == 255


def test_quality_preflight_keeps_normal_content():
    import numpy as np
    from zigvideo.quality import assess_tensor_video

    source = np.full((2, 3, 8, 8), 0.5, dtype=np.float32)
    frames, metrics = assess_tensor_video(source)
    assert len(frames) == 2
    assert metrics["status"] == "plausible"
    assert metrics["bright_pixel_fraction"] == 1.0



def test_latent_stats_remain_finite():
    import torch
    from zigvideo.quality import summarize_latents

    x = torch.tensor([[[1.0, 2.0, -3.0]]], dtype=torch.float16)
    stats = summarize_latents(x, step=3)
    assert stats["step"] == 3
    assert stats["nonfinite_fraction"] == 0
    assert stats["finite_min"] == -3.0
    assert stats["finite_max"] == 2.0


def test_latent_stats_detect_nan_and_inf():
    import torch
    from zigvideo.quality import summarize_latents

    x = torch.tensor([1.0, float("nan"), float("inf"), -2.0])
    stats = summarize_latents(x, step=2)
    assert stats["nonfinite_fraction"] == 0.5
    assert stats["finite_min"] == -2.0


def test_numerical_stage_diagnosis():
    from zigvideo.quality import classify_numerical_stage

    assert classify_numerical_stage(
        [{"nonfinite_fraction": 0.0}],
        {"nonfinite_fraction": 0.18},
    ) == "latents_finite_at_sampled_steps_but_postdecode_nonfinite"
    assert classify_numerical_stage(
        [{"nonfinite_fraction": 0.1}],
        {"nonfinite_fraction": 0.18},
    ) == "nonfinite_observed_during_denoising"



def test_cogvideox_fp32_vae_preview_keeps_transformer_fp16(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "preview.mp4",
        aspect="9:16",
        num_inference_steps=8,
        vae_fp32=True,
        save_latents=True,
    )
    assert preview["precision"] == "float16"
    assert preview["vae_precision"] == "float32"
    assert preview["save_latents"] is True
    assert preview["native_generation"]["num_inference_steps"] == 8


def test_raw_vae_check_detects_nonfinite_values():
    import torch
    from zigvideo.quality import summarize_decoded_tensor

    frames = torch.tensor(
        [[[[[0.25, float("nan"), float("inf")]]]]],
        dtype=torch.float16,
    )
    result = summarize_decoded_tensor(frames)
    assert result["dtype"] == "torch.float16"
    assert result["nonfinite_fraction"] == 0.66666667


def test_stage_classification_separates_raw_vae_from_postprocessor():
    from zigvideo.quality import identify_decode_stage

    latent_checks = [{"nonfinite_fraction": 0.0}]
    assert identify_decode_stage(
        latent_checks,
        {"nonfinite_fraction": 0.18},
        {"nonfinite_fraction": 0.18},
    ) == "nonfinite_in_raw_vae_output"
    assert identify_decode_stage(
        latent_checks,
        {"nonfinite_fraction": 0.0},
        {"nonfinite_fraction": 0.18},
    ) == "nonfinite_after_vae_before_or_during_postprocess"



def test_latent_safetensors_roundtrip(tmp_path):
    import torch
    from safetensors.torch import load_file, save_file

    latents = torch.ones((1, 4, 16, 8, 8), dtype=torch.float16)
    path = tmp_path / "result.latents.safetensors"
    save_file({"latents": latents}, str(path))
    loaded = load_file(str(path))["latents"]
    assert loaded.dtype == torch.float16
    assert loaded.shape == latents.shape
    assert torch.equal(loaded, latents)



def test_cogvideox_offload_group_preview_is_opt_in(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "group-smoke.mp4",
        aspect="9:16",
        num_inference_steps=1,
        offload_strategy="group",
    )
    assert preview["offload_strategy"] == "group"
    assert preview["native_generation"]["num_inference_steps"] == 1
    assert "experimental" in preview["offload"]


def test_cogvideox_default_offload_remains_sequential(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "baseline.mp4",
    )
    assert preview["offload_strategy"] == "sequential"


def test_cogvideox_group_offload_configures_components_separately(monkeypatch):
    import torch
    import diffusers.hooks as hooks

    from zigvideo.backends.cogvideox_fp16 import configure_offload

    calls = []

    class FakeModule:
        def __init__(self, name):
            self.name = name

        def enable_group_offload(self, **kwargs):
            calls.append((self.name, kwargs))

    class FakePipeline:
        def __init__(self):
            self.transformer = FakeModule("transformer")
            self.vae = FakeModule("vae")
            self.text_encoder = FakeModule("text_encoder")
            self.sequential_called = False
            self.pipeline_group_called = False

        def enable_sequential_cpu_offload(self, **kwargs):
            self.sequential_called = True

        def enable_group_offload(self, **kwargs):
            self.pipeline_group_called = True

    def capture_text_encoder_offload(module, **kwargs):
        calls.append((module.name, kwargs))

    monkeypatch.setattr(hooks, "apply_group_offloading", capture_text_encoder_offload)
    pipeline = FakePipeline()
    configure_offload(pipeline, torch, "group")

    assert not pipeline.sequential_called
    assert not pipeline.pipeline_group_called
    assert [name for name, _ in calls] == ["transformer", "vae", "text_encoder"]

    transformer = calls[0][1]
    assert transformer["onload_device"] == torch.device("cuda")
    assert transformer["offload_type"] == "block_level"
    assert transformer["num_blocks_per_group"] == 1
    assert transformer["use_stream"] is False

    for _, arguments in calls[1:]:
        assert arguments["offload_type"] == "leaf_level"
        assert arguments["use_stream"] is False


def test_cogvideox_sequential_strategy_does_not_apply_group_hooks():
    import torch

    from zigvideo.backends.cogvideox_fp16 import configure_offload

    class FakePipeline:
        def __init__(self):
            self.sequential_called = False

        def enable_sequential_cpu_offload(self, **kwargs):
            self.sequential_called = kwargs["device"] == "cuda"

    pipeline = FakePipeline()
    configure_offload(pipeline, torch, "sequential")
    assert pipeline.sequential_called


def test_cogvideox_invalid_offload_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="Unsupported offload"):
        generation_preview(
            preset="ultra-safe",
            cache_dir=tmp_path / "models",
            output=tmp_path / "invalid.mp4",
            offload_strategy="not-valid",
        )



def test_save_final_latents_before_decode(tmp_path):
    import torch
    from safetensors import safe_open

    from zigvideo.backends.cogvideox_fp16 import (
        CogVideoXAttempt,
        save_final_latents,
    )

    attempt = CogVideoXAttempt(720, 480, 16, 8, 1)
    latents = torch.ones((1, 4, 16, 60, 90), dtype=torch.float16)
    output = tmp_path / "one-step-smoke.mp4"
    saved_path = save_final_latents(latents, output, attempt, seed=42)
    assert saved_path.endswith("one-step-smoke.latents.safetensors")

    with safe_open(saved_path, framework="pt", device="cpu") as file:
        assert file.get_slice("latents").get_shape() == [1, 4, 16, 60, 90]
        assert file.metadata()["steps"] == "1"
        assert file.metadata()["seed"] == "42"
