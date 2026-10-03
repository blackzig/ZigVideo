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
