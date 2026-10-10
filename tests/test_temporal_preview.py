"""Cost-aware, opt-in 33-frame CogVideoX experiment (dry-run only)."""

import json

import pytest

from zigvideo.backends.cogvideox_fp16 import (
    generation_preview,
    validate_frame_override,
)
from zigvideo.cli import build_parser


def test_default_ultra_safe_keeps_16_frames_and_30_steps(tmp_path):
    preview = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "normal.mp4"
    )
    assert preview["native_generation"]["num_frames"] == 16
    assert preview["native_generation"]["num_inference_steps"] == 30
    assert preview["native_generation"]["guidance_scale"] == 6.0
    assert preview["source_duration_seconds"] == 2.0
    assert preview["temporal_experiment"] is None


def test_frame_override_keeps_all_other_ultra_safe_parameters(tmp_path):
    base = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "base.mp4"
    )
    extended = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "extended.mp4",
        num_frames=33,
    )
    old = base["native_generation"]
    new = extended["native_generation"]
    assert old["num_frames"] == 16
    assert new["num_frames"] == 33
    for field in ("width", "height", "fps", "num_inference_steps", "guidance_scale"):
        assert new[field] == old[field]
    assert extended["source_duration_seconds"] == 4.125
    assert extended["temporal_experiment"]["quality_validated"] is False
    assert extended["temporal_experiment"]["time_validated"] is False
    assert "nonlinearly" in extended["temporal_experiment"]["note"]
    assert extended["cfg_step_batch_factors"] is None


def test_safe_preset_is_unchanged_when_no_override(tmp_path):
    preview = generation_preview(
        "safe", tmp_path / "models", tmp_path / "safe.mp4"
    )
    assert preview["native_generation"]["num_frames"] == 33
    assert preview["native_generation"]["num_inference_steps"] == 40
    assert preview["temporal_experiment"] is None


@pytest.mark.parametrize("frames", [0, 15, 17, 20, 32, 50, -1, True])
def test_invalid_frame_override_rejected_before_pipeline_load(tmp_path, frames):
    with pytest.raises(ValueError, match="--frames"):
        generation_preview(
            "ultra-safe", tmp_path / "models", tmp_path / "invalid.mp4",
            num_frames=frames,
        )


@pytest.mark.parametrize("frames,seconds", [(16, 2.0), (33, 4.125), (49, 6.125)])
def test_allowed_frame_buckets_expose_unambiguous_source_duration(tmp_path, frames, seconds):
    validate_frame_override(frames)
    preview = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "frames.mp4",
        num_frames=frames,
    )
    assert preview["source_duration_seconds"] == seconds
    assert preview["native_generation"]["num_frames"] == frames


def test_33frame_robot_walk_dry_run_keeps_30step_cfg6_full(tmp_path, capsys):
    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--scene-profile", "robot-walk",
        "--preset", "ultra-safe", "--aspect", "9:16",
        "--frames", "33", "--steps", "30",
        "--seed", "42", "--cfg-scale", "6",
        "--offload", "sequential", "--latents-only",
        "--output", str(tmp_path / "walk-33.mp4"),
        "--dry-run",
    ])
    assert args.func(args) == 0
    preview = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert preview["scene_profile"] == "robot-walk"
    assert "left foot visibly lifts" in preview["prompt"]
    assert preview["native_generation"] == {
        "width": 720, "height": 480, "num_frames": 33,
        "fps": 8, "num_inference_steps": 30, "guidance_scale": 6.0,
    }
    assert preview["offload_strategy"] == "sequential"
    assert preview["cfg_guided_steps"] is None
    assert preview["source_duration_seconds"] == 4.125
    assert preview["latents_only"] is True
    assert preview["save_latents"] is True
    assert preview["temporal_experiment"]["quality_validated"] is False


def test_ltx_rejects_cogvideox_only_frame_override(tmp_path):
    args = build_parser().parse_args([
        "generate", "--backend", "ltx", "--prompt", "a robot",
        "--frames", "33", "--output", str(tmp_path / "invalid.mp4"),
        "--dry-run",
    ])
    with pytest.raises(ValueError, match="require the CogVideoX backend"):
        args.func(args)


def test_temporal_cfg_window_uses_step_count_not_frame_count(tmp_path):
    preview = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "selective.mp4",
        num_frames=33, num_inference_steps=30,
        cfg_scale=6.0, latents_only=True,
        cfg_guided_start=5, cfg_guided_steps=20,
    )
    assert preview["cfg_step_batch_factors"] == [1]*5 + [2]*20 + [1]*5
    assert preview["native_generation"]["num_frames"] == 33
