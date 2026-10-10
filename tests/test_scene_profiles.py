"""Scene profile composition and prompt-provenance tests (CPU-only)."""

import json

import pytest
import torch
from safetensors import safe_open

from zigvideo.scene_profiles import (
    DEFAULT_NEGATIVE_PROMPT,
    resolve_generation_text,
)


def test_ordinary_prompt_and_negative_prompt_are_unchanged():
    selected = resolve_generation_text("A friendly robot")
    assert selected.prompt == "A friendly robot"
    assert selected.negative_prompt == DEFAULT_NEGATIVE_PROMPT
    assert selected.scene_profile is None
    assert selected.quality_goal is None


def test_opt_in_robot_walk_places_subject_in_central_safe_zone_with_actions():
    selected = resolve_generation_text(None, "robot-walk")
    assert selected.scene_profile == "robot-walk"
    assert "complete robot" in selected.prompt
    assert "both feet" in selected.prompt
    assert "middle of the image" in selected.prompt
    assert "left foot visibly lifts" in selected.prompt
    assert "right foot lifts" in selected.prompt
    assert "stationary tripod camera" in selected.prompt
    assert "cars" in selected.negative_prompt
    assert "cropped legs" in selected.negative_prompt
    assert "Not guaranteed" in selected.quality_goal


@pytest.mark.parametrize(
    "prompt,profile",
    [
        (None, None),
        (" ", None),
        ("custom text", "robot-walk"),
        (None, "unknown-scene"),
    ],
)
def test_scene_resolver_rejects_missing_conflicting_or_unknown_input(prompt, profile):
    with pytest.raises(ValueError):
        resolve_generation_text(prompt, profile)


def test_explicit_negative_prompt_override_is_respected():
    selected = resolve_generation_text("A friendly robot", negative_prompt="No cars")
    assert selected.negative_prompt == "No cars"
    selected_profile = resolve_generation_text(
        None, "robot-walk", negative_prompt="No buildings"
    )
    assert selected_profile.negative_prompt == "No buildings"
    with pytest.raises(ValueError, match="negative-prompt"):
        resolve_generation_text("A friendly robot", negative_prompt="   ")


def test_scene_profile_dry_run_resolves_text_before_gpu(tmp_path, capsys):
    from zigvideo.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--scene-profile", "robot-walk", "--aspect", "9:16",
        "--output", str(tmp_path / "new-scene.mp4"),
        "--steps", "30", "--seed", "42", "--dry-run",
    ])
    assert args.func(args) == 0
    preview = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert preview["scene_profile"] == "robot-walk"
    assert "left foot visibly lifts" in preview["prompt"]
    assert "cropped legs" in preview["negative_prompt"]
    assert preview["scene_quality_goal"]
    assert preview["native_generation"]["num_inference_steps"] == 30
    assert preview["native_generation"]["guidance_scale"] == 6.0
    assert preview["output_mode"] == "mp4"


def test_old_prompt_dry_run_preserves_default_text(tmp_path, capsys):
    from zigvideo.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox", "--prompt", "old prompt",
        "--output", str(tmp_path / "original.mp4"), "--dry-run",
    ])
    assert args.func(args) == 0
    preview = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert preview["prompt"] == "old prompt"
    assert preview["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT
    assert preview["scene_profile"] is None
    assert preview["native_generation"]["guidance_scale"] == 6.0


def test_profile_is_not_silently_used_with_ltx(tmp_path):
    from zigvideo.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--backend", "ltx", "--scene-profile", "robot-walk",
        "--output", str(tmp_path / "invalid.mp4"), "--dry-run",
    ])
    with pytest.raises(ValueError, match="require --backend cogvideox"):
        args.func(args)


def test_profile_and_prompt_conflict_before_model_loading(tmp_path):
    from zigvideo.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--prompt", "another scene", "--scene-profile", "robot-walk",
        "--output", str(tmp_path / "invalid.mp4"), "--dry-run",
    ])
    with pytest.raises(ValueError, match="not both"):
        args.func(args)


def test_prompt_is_saved_with_latents_for_future_comparisons(tmp_path):
    from zigvideo.backends.cogvideox_fp16 import CogVideoXAttempt, save_final_latents

    path = save_final_latents(
        torch.zeros((1, 4, 16, 60, 90), dtype=torch.float16),
        tmp_path / "robot.mp4",
        CogVideoXAttempt(720, 480, 16, 8, 30, guidance_scale=6.0),
        42,
        prompt="Full robot in center",
        negative_prompt="No cars",
        scene_profile="robot-walk",
    )
    with safe_open(path, framework="pt", device="cpu") as sf:
        metadata = sf.metadata()
    assert metadata["prompt"] == "Full robot in center"
    assert metadata["negative_prompt"] == "No cars"
    assert metadata["scene_profile"] == "robot-walk"
    assert metadata["seed"] == "42"
    assert metadata["guidance_scale"] == "6.0"
