"""Experimental CogVideoX CFG step schedule without changing the Diffusers pipeline loop.

Diffusers CogVideoX v0.40.0 decides whether to use CFG only once before the
denoising loop. Changing its guidance_scale through a callback does NOT halve
transformer work. For selected steps we shorten the *actual* transformer input
to the positive/conditional half and duplicate its prediction on the way out,
so the upstream CFG combination evaluates to the conditional prediction.

Only use with the tested sequential-offload backend; not a general-purpose
patch for all schedulers, caches, or custom transformers.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field


def validate_selective_cfg(
    guided_steps: int | None,
    total_steps: int,
    guidance_scale: float,
    latents_only: bool,
    offload_strategy: str,
    guided_start: int = 0,
) -> None:
    if not isinstance(guided_start, int) or isinstance(guided_start, bool):
        raise ValueError("--cfg-guided-start must be an integer.")
    if guided_steps is None:
        if guided_start != 0:
            raise ValueError("--cfg-guided-start requires --cfg-guided-steps.")
        return
    if not isinstance(guided_steps, int) or isinstance(guided_steps, bool):
        raise ValueError("--cfg-guided-steps must be an integer.")
    if not 0 <= guided_steps <= total_steps:
        raise ValueError(
            "--cfg-guided-steps must be between 0 and --steps (inclusive)."
        )
    if guided_start < 0 or guided_start + guided_steps > total_steps:
        raise ValueError(
            "--cfg-guided-start and --cfg-guided-steps must fit within --steps."
        )
    if guidance_scale <= 1.0:
        raise ValueError("Selective CFG requires a baseline --cfg-scale > 1.")
    if not latents_only:
        raise ValueError(
            "Selective CFG is restricted to --latents-only until validated."
        )
    if offload_strategy != "sequential":
        raise ValueError(
            "Selective CFG currently requires --offload sequential."
        )



def cfg_step_batch_factors(
    total_steps: int, guided_steps: int, guided_start: int = 0
) -> list[int]:
    """Expected transformer batch factor per step for a guided window."""
    return [
        2 if guided_start <= i < guided_start + guided_steps else 1
        for i in range(total_steps)
    ]


@dataclass
class SelectiveCFGStats:
    requested_guided_steps: int
    total_steps: int
    requested_guided_start: int = 0
    calls_seen: int = 0
    guided_calls: int = 0
    conditional_only_calls: int = 0
    actual_transformer_batch_factors: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requested_guided_steps": self.requested_guided_steps,
            "requested_guided_start": self.requested_guided_start,
            "guided_step_window": [
                self.requested_guided_start,
                self.requested_guided_start + self.requested_guided_steps,
            ],
            "total_steps": self.total_steps,
            "calls_seen": self.calls_seen,
            "guided_calls": self.guided_calls,
            "conditional_only_calls": self.conditional_only_calls,
            "actual_transformer_batch_factors": list(
                self.actual_transformer_batch_factors
            ),
            "note": (
                "Only steps inside the requested zero-based half-open window "
                "use ordinary CFG. All other steps run the conditional "
                "transformer half only, and duplicate its prediction for "
                "compatibility with the upstream CFG math. "
                "Not a quality-validated mode."
            ),
        }


@contextmanager
def selective_cfg_transformer(
    transformer, *, guided_steps: int, total_steps: int, guided_start: int = 0
):
    """Temporarily install call hooks; restore even on pipeline failure.

    This assumes a single transformer call per denoising step, as in pinned
    diffusers CogVideoXPipeline 0.40.0. The hooks run at the module boundary,
    outside Accelerate's wrapped forward, so sequential hooks remain intact.
    """
    stats = SelectiveCFGStats(guided_steps, total_steps, guided_start)
    pending_conditional = False

    def prehook(_module, args, kwargs):
        nonlocal pending_conditional
        if stats.calls_seen >= total_steps:
            raise RuntimeError(
                "Selective CFG expected one transformer call per denoising step."
            )
        pending_conditional = not (
            guided_start <= stats.calls_seen < guided_start + guided_steps
        )
        stats.calls_seen += 1
        stats.actual_transformer_batch_factors.append(
            1 if pending_conditional else 2
        )
        if not pending_conditional:
            stats.guided_calls += 1
            return args, kwargs

        # Diffusers passes the concatenated [negative, positive] batch via
        # kwargs for hidden_states, encoder_hidden_states and timestep.
        # Never touch image_rotary_emb, attention_kwargs or return_dict.
        required = ("hidden_states", "encoder_hidden_states", "timestep")
        if args or any(name not in kwargs for name in required):
            raise RuntimeError(
                "Unsupported CogVideoX transformer call signature for selective CFG."
            )
        original_batch = kwargs["hidden_states"].shape[0]
        if original_batch % 2:
            raise RuntimeError("CFG transformer input batch must be even.")
        sliced = dict(kwargs)
        for name in required:
            value = kwargs[name]
            if value.shape[0] != original_batch:
                raise RuntimeError(f"Unexpected batch size in transformer {name}.")
            sliced[name] = value.chunk(2, dim=0)[1]
        stats.conditional_only_calls += 1
        return args, sliced

    def posthook(_module, _args, _kwargs, output):
        nonlocal pending_conditional
        if not pending_conditional:
            return output
        pending_conditional = False
        if not isinstance(output, tuple) or not output:
            raise RuntimeError("CogVideoX transformer must return a nonempty tuple.")
        import torch

        prediction = output[0]
        duplicated = torch.cat([prediction, prediction], dim=0)
        return (duplicated, *output[1:])

    pre_handle = transformer.register_forward_pre_hook(prehook, with_kwargs=True)
    try:
        post_handle = transformer.register_forward_hook(posthook, with_kwargs=True)
        try:
            yield stats
            if stats.calls_seen != total_steps:
                raise RuntimeError(
                    "Selective CFG transformer call count does not match "
                    f"steps: {stats.calls_seen} vs {total_steps}."
                )
        finally:
            post_handle.remove()
    finally:
        pre_handle.remove()
