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
