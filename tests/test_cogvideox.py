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
