import pytest
import torch

from zigvideo.selective_cfg import selective_cfg_transformer, validate_selective_cfg


class MiniTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, *, hidden_states, encoder_hidden_states, timestep, return_dict=False):
        self.batch_sizes.append(hidden_states.shape[0])
        prediction = (
            hidden_states
            + encoder_hidden_states
            + timestep[:, None].to(hidden_states.dtype)
        )
        return (prediction,)


def _inputs():
    return {
        "hidden_states": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "encoder_hidden_states": torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        "timestep": torch.tensor([2.0, 2.0]),
        "return_dict": False,
    }


def test_selective_cfg_only_halves_real_transformer_batch_after_guided_steps():
    transformer = MiniTransformer()
    kwargs = _inputs()
    with selective_cfg_transformer(
        transformer, guided_steps=1, total_steps=2
    ) as stats:
        guided_result = transformer(**kwargs)[0]
        unguided_result = transformer(**kwargs)[0]

    assert transformer.batch_sizes == [2, 1]
    assert guided_result.shape[0] == 2
    assert unguided_result.shape[0] == 2
    assert stats.to_dict()["actual_transformer_batch_factors"] == [2, 1]
    assert stats.guided_calls == 1
    assert stats.conditional_only_calls == 1


def test_conditional_only_reconstruction_preserves_upstream_cfg_math():
    transformer = MiniTransformer()
    kwargs = _inputs()
    baseline = transformer(**kwargs)[0]
    expected_conditional = baseline.chunk(2)[1]

    with selective_cfg_transformer(
        transformer, guided_steps=0, total_steps=1
    ):
        reconstruction = transformer(**kwargs)[0]

    reconstructed_uncond, reconstructed_cond = reconstruction.chunk(2)
    upstream_cfg_value = reconstructed_uncond + 6.0 * (
        reconstructed_cond - reconstructed_uncond
    )
    torch.testing.assert_close(upstream_cfg_value, expected_conditional)


def test_guided_steps_preserve_original_cfg_output():
    transformer = MiniTransformer()
    kwargs = _inputs()
    original = transformer(**kwargs)[0]
    with selective_cfg_transformer(
        transformer, guided_steps=2, total_steps=2
    ):
        first = transformer(**kwargs)[0]
        second = transformer(**kwargs)[0]
    torch.testing.assert_close(first, original)
    torch.testing.assert_close(second, original)
    assert transformer.batch_sizes == [2, 2, 2]


def test_cfg_hooks_are_removed_after_errors():
    transformer = MiniTransformer()
    with pytest.raises(RuntimeError, match="simulated"):
        with selective_cfg_transformer(
            transformer, guided_steps=0, total_steps=2
        ):
            transformer(**_inputs())
            raise RuntimeError("simulated")
    assert len(transformer._forward_pre_hooks) == 0
    assert len(transformer._forward_hooks) == 0


def test_cfg_hooks_detect_missing_calls_and_still_remove():
    transformer = MiniTransformer()
    with pytest.raises(RuntimeError, match="call count"):
        with selective_cfg_transformer(
            transformer, guided_steps=1, total_steps=2
        ):
            transformer(**_inputs())
    assert len(transformer._forward_pre_hooks) == 0
    assert len(transformer._forward_hooks) == 0


@pytest.mark.parametrize(
    "guided,total,cfg,latents,offload",
    [
        (3, 2, 6.0, True, "sequential"),
        (-1, 2, 6.0, True, "sequential"),
        (True, 2, 6.0, True, "sequential"),
        (1, 2, 1.0, True, "sequential"),
        (1, 2, 6.0, False, "sequential"),
        (1, 2, 6.0, True, "group"),
    ],
)
def test_cfg_schedule_validates_safety(guided, total, cfg, latents, offload):
    with pytest.raises(ValueError):
        validate_selective_cfg(guided, total, cfg, latents, offload)


def test_cfg_schedule_disabled_does_not_enforce_latents_only():
    validate_selective_cfg(None, 30, 6.0, False, "sequential")


def test_cfg_schedule_preview_shows_exact_step_batch_factors(tmp_path):
    from zigvideo.backends.cogvideox_fp16 import generation_preview

    preview = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "output.mp4",
        num_inference_steps=4, cfg_scale=6.0,
        latents_only=True, offload_strategy="sequential",
        cfg_guided_steps=2,
    )
    assert preview["cfg_guided_steps"] == 2
    assert preview["cfg_step_batch_factors"] == [2, 2, 1, 1]
    assert preview["native_generation"]["guidance_scale"] == 6.0


def test_cli_exposes_schedule_in_dry_run(tmp_path, capsys):
    import json
    from zigvideo.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "generate", "--backend", "cogvideox", "--prompt", "robot",
        "--output", str(tmp_path / "out.mp4"),
        "--steps", "2", "--cfg-scale", "6", "--latents-only",
        "--cfg-guided-steps", "1", "--dry-run"
    ])
    assert args.func(args) == 0
    result = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert result["cfg_step_batch_factors"] == [2, 1]


def test_middle_cfg_window_reduces_real_batches_before_and_after():
    transformer = MiniTransformer()
    with selective_cfg_transformer(
        transformer, guided_steps=1, guided_start=1, total_steps=3
    ) as stats:
        first = transformer(**_inputs())[0]
        second = transformer(**_inputs())[0]
        third = transformer(**_inputs())[0]

    assert transformer.batch_sizes == [1, 2, 1]
    assert all(x.shape[0] == 2 for x in (first, second, third))
    assert stats.guided_calls == 1
    assert stats.conditional_only_calls == 2
    assert stats.to_dict()["requested_guided_start"] == 1
    assert stats.to_dict()["guided_step_window"] == [1, 2]
    assert stats.to_dict()["actual_transformer_batch_factors"] == [1, 2, 1]
    assert len(transformer._forward_hooks) == 0


@pytest.mark.parametrize(
    "guided,start,total",
    [(2, 2, 3), (1, -1, 3), (1, 4, 3), (1, True, 3)],
)
def test_cfg_window_rejects_invalid_ranges(guided, start, total):
    with pytest.raises(ValueError):
        validate_selective_cfg(
            guided, total, 6.0, True, "sequential", guided_start=start
        )


def test_nonzero_start_requires_guided_step_count():
    with pytest.raises(ValueError, match="requires"):
        validate_selective_cfg(
            None, 30, 6.0, True, "sequential", guided_start=8
        )


def test_middle_window_preview_and_cli(tmp_path, capsys):
    import json

    from zigvideo.backends.cogvideox_fp16 import generation_preview
    from zigvideo.cli import build_parser

    preview = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "mid.mp4",
        num_inference_steps=5, cfg_scale=6.0, latents_only=True,
        offload_strategy="sequential", cfg_guided_steps=2,
        cfg_guided_start=2,
    )
    assert preview["cfg_guided_start"] == 2
    assert preview["cfg_step_batch_factors"] == [1, 1, 2, 2, 1]

    parser = build_parser()
    args = parser.parse_args([
        "generate", "--backend", "cogvideox", "--prompt", "robot",
        "--steps", "3", "--cfg-scale", "6", "--latents-only",
        "--cfg-guided-steps", "1", "--cfg-guided-start", "1",
        "--dry-run",
    ])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert payload["cfg_step_batch_factors"] == [1, 2, 1]
    assert payload["cfg_guided_start"] == 1


def test_latent_metadata_distinguishes_window_cfg_from_constant_cfg(tmp_path):
    from safetensors import safe_open
    from zigvideo.backends.cogvideox_fp16 import (
        CogVideoXAttempt,
        save_final_latents,
    )

    attempt = CogVideoXAttempt(720, 480, 16, 8, 3, guidance_scale=6.0)
    path = save_final_latents(
        torch.zeros((1, 4, 16, 60, 90), dtype=torch.float16),
        tmp_path / "window.mp4", attempt, 42,
        cfg_guided_steps=1, cfg_guided_start=1,
    )
    with safe_open(path, framework="pt", device="cpu") as file:
        meta = file.metadata()
    assert meta["guidance_scale"] == "6.0"
    assert meta["cfg_schedule"] == "selective_window"
    assert meta["cfg_guided_steps"] == "1"
    assert meta["cfg_guided_start"] == "1"


def test_experimental_cfg_mp4_requires_explicit_opt_in():
    with pytest.raises(ValueError, match="experimental-cfg-video"):
        validate_selective_cfg(
            20, 30, 6.0, False, "sequential", guided_start=5
        )


def test_experimental_cfg_video_requires_mixed_window_and_steps():
    with pytest.raises(ValueError, match="requires --cfg-guided-steps"):
        validate_selective_cfg(
            None, 30, 6.0, False, "sequential",
            experimental_video=True,
        )
    with pytest.raises(ValueError, match="mixed"):
        validate_selective_cfg(
            30, 30, 6.0, False, "sequential",
            experimental_video=True,
        )
    with pytest.raises(ValueError, match="mixed"):
        validate_selective_cfg(
            0, 30, 6.0, False, "sequential",
            experimental_video=True,
        )


def test_experimental_cfg_video_rejects_conflicting_latent_mode():
    with pytest.raises(ValueError, match="cannot use --latents-only"):
        validate_selective_cfg(
            20, 30, 6.0, True, "sequential",
            guided_start=5, experimental_video=True,
        )


def test_selective_cfg_mp4_preview_forces_saved_latents(tmp_path):
    from zigvideo.backends.cogvideox_fp16 import generation_preview

    preview = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "quality.mp4",
        num_inference_steps=30, cfg_scale=6.0,
        cfg_guided_steps=20, cfg_guided_start=5,
        latents_only=False,
        experimental_cfg_video=True,
    )
    assert preview["latents_only"] is False
    assert preview["experimental_cfg_video"] is True
    assert preview["output_mode"] == "experimental_cfg_mp4"
    assert preview["save_latents"] is True
    assert preview["cfg_step_batch_factors"] == [1] * 5 + [2] * 20 + [1] * 5


def test_normal_cogvideo_generation_dry_run_is_unaffected(tmp_path):
    from zigvideo.backends.cogvideox_fp16 import generation_preview

    normal = generation_preview(
        "ultra-safe", tmp_path / "models", tmp_path / "normal.mp4"
    )
    assert normal["output_mode"] == "mp4"
    assert normal["save_latents"] is False
    assert normal["cfg_guided_steps"] is None
    assert normal["experimental_cfg_video"] is False
    assert normal["native_generation"]["guidance_scale"] == 6.0


def test_cli_experimental_cfg_video_dry_run(tmp_path, capsys):
    import json

    from zigvideo.cli import build_parser

    parser = build_parser()
    args = parser.parse_args([
        "generate", "--backend", "cogvideox",
        "--prompt", "A small friendly robot walking through a rainy futuristic city",
        "--output", str(tmp_path / "selective.mp4"),
        "--steps", "30", "--cfg-scale", "6",
        "--cfg-guided-start", "5", "--cfg-guided-steps", "20",
        "--experimental-cfg-video", "--dry-run",
    ])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert payload["output_mode"] == "experimental_cfg_mp4"
    assert payload["save_latents"] is True
    assert payload["cfg_step_batch_factors"] == [1] * 5 + [2] * 20 + [1] * 5


def test_video_schedule_rejects_group_offload_even_with_explicit_flag():
    with pytest.raises(ValueError, match="sequential"):
        validate_selective_cfg(
            20, 30, 6.0, False, "group",
            guided_start=5, experimental_video=True,
        )
