"""Token-limit safety checks. All tests are CPU-only; no model downloads."""

import json

import pytest
import torch

from zigvideo.prompt_preflight import (
    COGVIDEOX_MAX_TEXT_TOKENS,
    count_prompt_tokens,
    validate_cogvideox_prompt_tokens,
)


class FakeT5Tokenizer:
    def __init__(self, counts):
        self.counts = counts
        self.calls = []

    def __call__(self, text, *, add_special_tokens, truncation, padding):
        self.calls.append((text, add_special_tokens, truncation, padding))
        return {"input_ids": list(range(self.counts[text]))}


def test_preflight_counts_special_tokens_without_truncation():
    tokenizer = FakeT5Tokenizer({"positive": 226, "negative": 225})
    report = validate_cogvideox_prompt_tokens(
        "positive", "negative", tokenizer=tokenizer,
    )
    assert COGVIDEOX_MAX_TEXT_TOKENS == 226
    assert report["prompt_tokens"] == 226
    assert report["negative_prompt_tokens"] == 225
    assert report["truncated"] is False
    assert tokenizer.calls == [
        ("positive", True, False, False),
        ("negative", True, False, False),
    ]


@pytest.mark.parametrize("counts,fragment", [
    ({"positive": 227, "negative": 200}, "prompt=227"),
    ({"positive": 200, "negative": 227}, "negative_prompt=227"),
    ({"positive": 241, "negative": 235}, "prompt=241"),
])
def test_preflight_rejects_both_types_of_truncation(counts, fragment):
    tokenizer = FakeT5Tokenizer(counts)
    with pytest.raises(ValueError, match=fragment):
        validate_cogvideox_prompt_tokens(
            "positive", "negative", tokenizer=tokenizer,
        )


def test_preflight_rejects_empty_prompts():
    tokenizer = FakeT5Tokenizer({"ok": 20})
    with pytest.raises(ValueError, match="nonempty"):
        validate_cogvideox_prompt_tokens("  ", "ok", tokenizer=tokenizer)


def test_preflight_rejects_unexpected_batch_tokenization_shape():
    class BatchedTokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [[4, 5, 6]]}

    with pytest.raises(ValueError, match="single"):
        count_prompt_tokens("some text", BatchedTokenizer())


def test_local_tokenizer_load_fails_closed_with_instruction(monkeypatch):
    from zigvideo import prompt_preflight

    class FakeT5:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"] is True
            raise OSError("tokenizer cache missing")

    import transformers
    monkeypatch.setattr(transformers, "T5Tokenizer", FakeT5)
    with pytest.raises(RuntimeError, match="no generation started"):
        prompt_preflight.load_cached_cogvideox_tokenizer("fake-cache")


def test_legacy_dry_run_skips_new_tokenizer_check(tmp_path, capsys, monkeypatch):
    from zigvideo.cli import build_parser
    from zigvideo import prompt_preflight

    def fail(*_args, **_kwargs):
        raise AssertionError("Unexpected tokenizer access in legacy dry-run")

    monkeypatch.setattr(prompt_preflight, "validate_cogvideox_prompt_tokens", fail)
    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--scene-profile", "robot-walk-lit-wide", "--dry-run",
        "--output", str(tmp_path / "old.mp4"),
    ])
    assert args.func(args) == 0
    output = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert "prompt_token_preflight" not in output


def test_token_checked_dry_run_refuses_oversize_before_gpu(tmp_path, monkeypatch):
    from zigvideo.cli import build_parser
    from zigvideo import prompt_preflight

    def fail(*_args, **_kwargs):
        raise ValueError("CogVideoX prompt would be truncated: prompt=241 tokens")

    monkeypatch.setattr(prompt_preflight, "validate_cogvideox_prompt_tokens", fail)
    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--scene-profile", "robot-walk-lit-wide",
        "--validate-prompt-tokens", "--dry-run",
        "--output", str(tmp_path / "too-long.mp4"),
    ])
    with pytest.raises(ValueError, match="prompt=241"):
        args.func(args)


def test_token_checked_dry_run_surfaces_verified_count(tmp_path, capsys, monkeypatch):
    from zigvideo.cli import build_parser
    from zigvideo import prompt_preflight

    checked = {
        "tokenizer": "CogVideoX-2b T5Tokenizer (cached)",
        "max_sequence_length": 226,
        "prompt_tokens": 155,
        "negative_prompt_tokens": 117,
        "truncated": False,
    }
    monkeypatch.setattr(
        prompt_preflight, "validate_cogvideox_prompt_tokens",
        lambda *_args, **_kwargs: checked,
    )
    args = build_parser().parse_args([
        "generate", "--backend", "cogvideox",
        "--scene-profile", "robot-walk-lit", "--dry-run",
        "--validate-prompt-tokens",
        "--output", str(tmp_path / "checked.mp4"),
    ])
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert report["prompt_token_preflight"] == checked
    assert report["native_generation"]["num_frames"] == 16


def test_token_check_dry_run_rejected_for_ltx(tmp_path):
    from zigvideo.cli import build_parser

    args = build_parser().parse_args([
        "generate", "--backend", "ltx", "--prompt", "hello",
        "--validate-prompt-tokens", "--dry-run",
        "--output", str(tmp_path / "other.mp4"),
    ])
    with pytest.raises(ValueError, match="requires the CogVideoX backend"):
        args.func(args)


def test_real_generation_rejects_tokens_before_loading_weights(tmp_path, monkeypatch):
    import zigvideo.backends.cogvideox_fp16 as backend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail(*_args, **_kwargs):
        raise ValueError("CogVideoX prompt would be truncated: prompt=241")

    def unexpected_pipeline(*_args, **_kwargs):
        raise AssertionError("Full video weights were loaded before prompt check!")

    monkeypatch.setattr(backend, "validate_cogvideox_prompt_tokens", fail)
    monkeypatch.setattr(backend, "_load_pipeline", unexpected_pipeline)
    with pytest.raises(ValueError, match="prompt=241"):
        backend.generate_text_to_video(
            prompt="too many tokens",
            output=tmp_path / "no-gpu.mp4",
            negative_prompt="no distortions",
            preset="ultra-safe",
            num_inference_steps=30,
            latents_only=True,
        )
