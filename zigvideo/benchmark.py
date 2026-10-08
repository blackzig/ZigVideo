from __future__ import annotations


def summarize_stage_timings(
    step_end_elapsed_seconds: list[float],
    vae_decode_seconds: float,
    pipeline_seconds: float,
) -> dict:
    """Summarize elapsed stages without claiming pipeline time equals denoising.

    The first step includes T5 prompt processing and scheduler setup.
    Intervals from step 2 onward are closer to steady denoising costs.
    """
    if not step_end_elapsed_seconds:
        raise ValueError("At least one denoising-step checkpoint is required")
    if any(b < a for a, b in zip(step_end_elapsed_seconds, step_end_elapsed_seconds[1:])):
        raise ValueError("Step checkpoints must be monotonically increasing")
    first = step_end_elapsed_seconds[0]
    denoised_at = step_end_elapsed_seconds[-1]
    if min(first, vae_decode_seconds, pipeline_seconds) < 0:
        raise ValueError("Negative duration supplied")
    if denoised_at > pipeline_seconds + 0.001:
        raise ValueError("Final step time cannot exceed pipeline time")

    step_intervals = [
        round(b - a, 3)
        for a, b in zip(step_end_elapsed_seconds, step_end_elapsed_seconds[1:])
    ]
    remaining = max(0.0, pipeline_seconds - denoised_at - vae_decode_seconds)

    return {
        "steps_observed": len(step_end_elapsed_seconds),
        "prompt_setup_and_first_step_seconds": round(first, 3),
        "subsequent_step_seconds": step_intervals,
        "subsequent_step_mean_seconds": (
            round(sum(step_intervals) / len(step_intervals), 3)
            if step_intervals
            else None
        ),
        "time_to_final_latents_seconds": round(denoised_at, 3),
        "vae_decode_seconds": round(vae_decode_seconds, 3),
        "remaining_pipeline_seconds": round(remaining, 3),
        "pipeline_total_seconds": round(pipeline_seconds, 3),
        "note": (
            "First checkpoint includes text encoding and setup. Subsequent "
            "step intervals approximate steady denoising. Remaining pipeline "
            "time includes postprocess, callbacks and hook cleanup."
        ),
    }
