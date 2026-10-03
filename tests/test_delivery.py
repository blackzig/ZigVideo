from zigvideo.delivery import build_video_filter


def test_motion_delivery_builds_24fps_720p_vertical_filter():
    value = build_video_filter(
        target_fps=24,
        target_width=720,
        target_height=1280,
        interpolation="motion",
    )
    assert "minterpolate=fps=24" in value
    assert "scale=720:1280" in value


def test_duplicate_delivery_uses_constant_fps_filter():
    value = build_video_filter(
        target_fps=30,
        target_width=720,
        target_height=1280,
        interpolation="duplicate",
    )
    assert value.startswith("fps=30,")
    assert "scale=720:1280" in value


def test_motion_delivery_preserves_source_duration():
    value = build_video_filter(
        target_fps=24,
        target_width=720,
        target_height=1280,
        interpolation="motion",
        source_fps=8.0,
        source_duration=2.0,
    )
    assert "tpad=stop_mode=clone:stop_duration=0.250000" in value
    assert "minterpolate=fps=24" in value
    assert "trim=duration=2.000000" in value
