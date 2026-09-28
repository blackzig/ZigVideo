from zigvideo.backends.ltx_gguf import (
    EMERGENCY_ATTEMPTS,
    PRESETS,
    QUANTIZATION_LABEL,
    SUPPORTED_ASPECTS,
    build_attempt_ladder,
    generation_preview,
)


def test_ultra_safe_uses_ltx_512_bucket():
    attempt = PRESETS["ultra-safe"]
    attempt.validate()
    assert attempt.width == 640
    assert attempt.height == 384
    assert attempt.num_frames == 9
    assert attempt.fps == 8
    assert attempt.num_inference_steps == 8


def test_fallback_ladder_preserves_minimum_useful_resolution():
    attempts = build_attempt_ladder("ultra-safe")
    assert attempts[0] == PRESETS["ultra-safe"]
    assert attempts[-1].width >= 512
    assert attempts[-1].height >= 320
    assert all((a.num_frames - 1) % 8 == 0 for a in attempts)


def test_generation_preview_does_not_load_models(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "first.mp4",
    )
    assert preview["backend"] == "ltx-gguf"
    assert preview["compute_dtype"] == "float16"
    assert preview["quantization"] == QUANTIZATION_LABEL
    assert preview["quantization"] == "GGUF Q5_K_M"
    assert preview["scheduler"] == "LTX FlowMatch + stochastic sampling"
    assert "staged whole-VAE decode" in preview["offload"]
    assert (
        preview["transformer_loader"]
        == "LTXVideoTransformer3DModel.from_single_file"
    )
    assert preview["attempts"][0]["width"] == 640
    assert preview["attempts"][0]["height"] == 384


def test_installed_diffusers_exposes_ltx_single_file_loader():
    from diffusers import LTXVideoTransformer3DModel

    assert hasattr(LTXVideoTransformer3DModel, "from_single_file")


def test_installed_diffusers_exposes_group_offload():
    from diffusers import LTXVideoTransformer3DModel
    from diffusers.hooks import apply_group_offloading

    assert callable(apply_group_offloading)
    assert hasattr(LTXVideoTransformer3DModel, "enable_group_offload")


def test_vertical_shorts_uses_native_9_16_bucket():
    attempts = build_attempt_ladder("ultra-safe", "9:16")
    first = attempts[0]
    assert first.width == 384
    assert first.height == 640
    assert first.width < first.height
    assert "9:16" in SUPPORTED_ASPECTS


def test_vertical_preview_recommends_short_delivery_resolution(tmp_path):
    preview = generation_preview(
        preset="ultra-safe",
        cache_dir=tmp_path / "models",
        output=tmp_path / "short.mp4",
        aspect="9:16",
    )
    assert preview["aspect"] == "9:16"
    assert preview["attempts"][0]["width"] == 384
    assert preview["attempts"][0]["height"] == 640
    assert "720x1280" in preview["recommended_delivery"]
