"""Fail-fast CogVideoX prompt token checks without loading video model weights.

The upstream Diffusers CogVideoX T5 encoder uses a max text sequence of 226
tokens (special tokens included). These checks use the same cached T5 tokenizer,
not a word count, to avoid silently dropping a prompt suffix during a costly run.
"""

from __future__ import annotations

from pathlib import Path

COGVIDEOX_MAX_TEXT_TOKENS = 226
COGVIDEOX_REPO = "THUDM/CogVideoX-2b"


def load_cached_cogvideox_tokenizer(cache_dir: str | Path):
    """Load only local cached tokenizer files; never download model weights."""
    from transformers import T5Tokenizer

    try:
        return T5Tokenizer.from_pretrained(
            COGVIDEOX_REPO,
            subfolder="tokenizer",
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    except OSError as exc:
        raise RuntimeError(
            "Could not open cached CogVideoX T5 tokenizer for safe prompt "
            "validation. The prompt may be truncated; no generation started. "
            "Verify --cache-dir and the local THUDM/CogVideoX-2b tokenizer cache."
        ) from exc


def count_prompt_tokens(text: str, tokenizer) -> int:
    """Use the same T5 special-token policy as CogVideoX prompt encoding."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Prompt text must be nonempty.")
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    ids = encoded["input_ids"]
    if not isinstance(ids, (list, tuple)) or (
        ids and isinstance(ids[0], (list, tuple))
    ):
        raise ValueError("Expected a single CogVideoX T5 token ID sequence.")
    return len(ids)


def validate_cogvideox_prompt_tokens(
    prompt: str,
    negative_prompt: str,
    cache_dir: str | Path = "models/huggingface",
    tokenizer=None,
) -> dict:
    """Reject either prompt if CogVideoX would truncate it at 226 tokens.

    Tokenizer injection is for local, CPU-only unit tests; production loads
    the cached tokenizer but does not load the diffusion model or VAE.
    """
    if tokenizer is None:
        tokenizer = load_cached_cogvideox_tokenizer(cache_dir)

    counts = {
        "prompt": count_prompt_tokens(prompt, tokenizer),
        "negative_prompt": count_prompt_tokens(negative_prompt, tokenizer),
    }
    too_long = [
        f"{name}={size}"
        for name, size in counts.items()
        if size > COGVIDEOX_MAX_TEXT_TOKENS
    ]
    if too_long:
        raise ValueError(
            "CogVideoX prompt would be truncated: "
            + ", ".join(too_long)
            + f" tokens (maximum {COGVIDEOX_MAX_TEXT_TOKENS}, including T5 "
            "special tokens). Shorten the prompt and re-run preflight; "
            "no video model was loaded."
        )

    return {
        "tokenizer": "CogVideoX-2b T5Tokenizer (cached)",
        "max_sequence_length": COGVIDEOX_MAX_TEXT_TOKENS,
        "prompt_tokens": counts["prompt"],
        "negative_prompt_tokens": counts["negative_prompt"],
        "truncated": False,
    }
