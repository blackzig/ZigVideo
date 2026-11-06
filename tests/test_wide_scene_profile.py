"""CPU-only checks for the opt-in robot-walk-lit-wide framing experiment."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors import safe_open

from zigvideo.scene_profiles import (
    DEFAULT_NEGATIVE_PROMPT,
    SCENE_PROFILES,
    _ROBOT_WALK_LIT_OPENING,
    _ROBOT_WALK_LIT_WIDE_OPENING,
    _ROBOT_WALK_LIT_FRAMING,
    _ROBOT_WALK_LIT_WIDE_FRAMING,
    resolve_generation_text,
)
from zigvideo.cli import build_parser


def test_wide_prompt_is_reversible_two_part_framing_edit_of_lit_profile():
    lit = SCENE_PROFILES["robot-walk-lit"]
    wide = SCENE_PROFILES["robot-walk-lit-wide"]
    restored = (
        wide.prompt
        .replace(_ROBOT_WALK_LIT_WIDE_OPENING, _ROBOT_WALK_LIT_OPENING)
        .replace(_ROBOT_WALK_LIT_WIDE_FRAMING, _ROBOT_WALK_LIT_FRAMING)
    )
    assert restored == lit.prompt
    assert "silver metallic body panels" in wide.prompt
    assert "two blue LED eyes" in wide.prompt
    assert "diffused frontal key light" in wide.prompt
    assert "left foot visibly lifts" in wide.prompt
    assert "right foot lifts" in wide.prompt
    assert "stationary tripod camera" in wide.prompt
    assert "portrait" not in wide.goal or "crop margins" in wide.goal


def test_wide_instructions_are_early_explicit_and_keep_original_profiles():
    original = SCENE_PROFILES["robot-walk"]
    lit = SCENE_PROFILES["robot-walk-lit"]
    wide = SCENE_PROFILES["robot-walk-lit-wide"]
    assert original.prompt.startswith("A single small friendly humanoid robot walks")
    assert lit.prompt.startswith("A single small friendly humanoid robot under")
    assert wide.prompt.startswith("A wide static camera shot")
    assert "half the picture height" in wide.prompt
    assert "strip of wet pavement below the shoes" in wide.prompt
    assert "centered precisely in the middle of the image" in wide.prompt
    assert "modest in size" in wide.prompt
    assert wide.negative_prompt.startswith(lit.negative_prompt)
    for exclusion in ("oversized robot", "feet touching the bottom edge", "missing headroom"):
        assert exclusion in wide.negative_prompt
    assert lit.negative_prompt.startswith(original.negative_prompt)
    assert original.negative_prompt.endswith(DEFAULT_NEGATIVE_PROMPT)


def _run_dry_run(args, capsys):
    parser = build_parser()
    parsed = parser.parse_args(args)
    assert parsed.func(parsed) == 0
    return json.loads(capsys.readouterr().out.split("\n", 1)[1])


def test_dry_run_wide_differs_from_lit_only_in_profile_and_text(tmp_path, capsys):
    flags = [
        "generate", "--backend", "cogvideox",
        "--preset", "ultra-safe", "--frames", "16",
        "--aspect", "9:16", "--steps", "30",
        "--seed", "42", "--offload", "sequential",
        "--cfg-scale", "6", "--latents-only", "--dry-run",
    ]
    lit = _run_dry_run(
        flags + ["--scene-profile", "robot-walk-lit",
                 "--output", str(tmp_path / "lit.mp4")], capsys,
    )
    wide = _run_dry_run(
        flags + ["--scene-profile", "robot-walk-lit-wide",
                 "--output", str(tmp_path / "wide.mp4")], capsys,
    )
    assert wide["scene_profile"] == "robot-walk-lit-wide"
    assert wide["native_generation"] == lit["native_generation"]
    assert wide["native_generation"] == {
        "width": 720, "height": 480, "num_frames": 16, "fps": 8,
        "num_inference_steps": 30, "guidance_scale": 6.0,
    }
    assert wide["latents_only"] is True
    assert wide["save_latents"] is True
    assert wide["cfg_guided_steps"] is None
    assert wide["cfg_step_batch_factors"] is None
    assert wide["experimental_cfg_video"] is False
    assert wide["offload_strategy"] == "sequential"
    assert wide["source_duration_seconds"] == 2.0
    assert wide["negative_prompt"] != lit["negative_prompt"]
    assert wide["prompt"] != lit["prompt"]
    assert "frontal key light" in wide["prompt"]
    assert "half the picture height" in wide["prompt"]
    assert wide["scene_quality_goal"]


def test_wide_profile_refuses_custom_prompt_conflict(tmp_path):
    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--scene-profile", "robot-walk-lit-wide",
        "--prompt", "a second prompt",
        "--output", str(tmp_path / "bad.mp4"), "--dry-run",
    ])
    with pytest.raises(ValueError, match="not both"):
        args.func(args)


def test_wide_profile_refuses_ltx_backend(tmp_path):
    args = build_parser().parse_args([
        "generate", "--backend", "ltx",
        "--scene-profile", "robot-walk-lit-wide",
        "--output", str(tmp_path / "bad.mp4"), "--dry-run",
    ])
    with pytest.raises(ValueError, match="require --backend cogvideox"):
        args.func(args)


def test_wide_profile_saves_exact_text_and_cfg_in_latent_metadata(tmp_path):
    from zigvideo.backends.cogvideox_fp16 import CogVideoXAttempt, save_final_latents

    profile = resolve_generation_text(None, "robot-walk-lit-wide")
    saved = save_final_latents(
        torch.zeros((1, 4, 16, 60, 90), dtype=torch.float16),
        tmp_path / "robot-walk-lit-wide.mp4",
        CogVideoXAttempt(720, 480, 16, 8, 30, 6.0),
        seed=42,
        prompt=profile.prompt,
        negative_prompt=profile.negative_prompt,
        scene_profile=profile.scene_profile,
    )
    with safe_open(saved, framework="pt", device="cpu") as file:
        metadata = file.metadata()
    assert metadata["scene_profile"] == "robot-walk-lit-wide"
    assert metadata["prompt"] == profile.prompt
    assert metadata["negative_prompt"] == profile.negative_prompt
    assert metadata["frames"] == "16"
    assert metadata["steps"] == "30"
    assert metadata["seed"] == "42"
    assert metadata["guidance_scale"] == "6.0"


def test_wide_explicit_negative_prompt_override_preserves_positive_prompt():
    before = resolve_generation_text(None, "robot-walk-lit-wide")
    after = resolve_generation_text(
        None, "robot-walk-lit-wide", negative_prompt="no pink sky",
    )
    assert after.prompt == before.prompt
    assert after.negative_prompt == "no pink sky"
    assert after.scene_profile == "robot-walk-lit-wide"
