from zigvideo.backends.ltx_gguf import (
    EMERGENCY_ATTEMPTS,
    PRESETS,
    build_attempt_ladder,
    generation_preview,
)


def test_ultra_safe_ltx_dimensions_are_valid():
    attempt = PRESETS["ultra-safe"]
    attempt.validate()
    assert attempt.width == 320
    assert attempt.height == 192
    assert attempt.num_frames == 9
    assert attempt.num_inference_steps == 8


def test_fallback_ladder_gets_smaller():
    attempts = build_attempt_ladder("ultra-safe")
    assert attempts[0] == PRESETS["ultra-safe"]
    assert attempts[-1].width <= attempts[0].width
    assert attempts[-1].height <= attempts[0].height
    assert all((a.num_frames - 1) % 8 == 0 for a in attempts)


def test_generation_preview_does_not_load_models(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "first.mp4",
    )
    assert preview["backend"] == "ltx-gguf"
    assert preview["compute_dtype"] == "float16"
    assert preview["quantization"] == "GGUF Q3_K_S"
    assert (
        preview["transformer_loader"]
        == "LTXVideoTransformer3DModel.from_single_file"
    )
    assert preview["attempts"][0]["num_frames"] == 9


def test_installed_diffusers_exposes_ltx_single_file_loader():
    from diffusers import LTXVideoTransformer3DModel

    assert hasattr(LTXVideoTransformer3DModel, "from_single_file")
