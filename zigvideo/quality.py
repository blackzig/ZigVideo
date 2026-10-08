from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image


def assess_tensor_video(video: Any) -> tuple[list[Image.Image], dict]:
    """Check normalized [frames, 3, height, width] tensor before uint8 export.

    This checks numerical integrity and luminance, not artistic quality.
    """
    if hasattr(video, "detach"):
        # Keep torch optional for non-inference users.
        video = video.detach().float().cpu().numpy()
    else:
        video = np.asarray(video, dtype=np.float32)

    if video.ndim != 4 or video.shape[1] != 3 or video.shape[0] < 1:
        raise ValueError("Expected RGB frames with shape [frames, 3, height, width].")

    finite = np.isfinite(video)
    nonfinite_fraction = float(1.0 - finite.mean())

    # Avoid silent NumPy cast warnings when a decoder produced NaN/Inf.
    clean = np.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0)
    clean = np.clip(clean, 0.0, 1.0)
    pixels = np.rint(np.transpose(clean, (0, 2, 3, 1)) * 255).astype(np.uint8)

    maximum_channel = pixels.max(axis=-1)
    maximum_value = int(pixels.max())
    bright_fraction = float(np.mean(maximum_channel > 32))
    near_black_fraction = float(np.mean(maximum_channel <= 3))

    if nonfinite_fraction > 0:
        status = "nonfinite_detected"
    elif maximum_value <= 3:
        status = "near_black"
    elif bright_fraction < 0.05:
        status = "very_dark"
    else:
        status = "plausible"

    metrics = {
        "status": status,
        "nonfinite_fraction": round(nonfinite_fraction, 8),
        "mean_pixel_value_0_255": round(float(pixels.mean()), 3),
        "maximum_pixel_value_0_255": maximum_value,
        "near_black_pixel_fraction": round(near_black_fraction, 6),
        "bright_pixel_fraction": round(bright_fraction, 6),
        "frames": int(pixels.shape[0]),
        "note": (
            "Numerical sanity check only; plausible does not guarantee visual "
            "quality. Non-finite pixels are sanitized for export."
        ),
    }
    return [Image.fromarray(frame) for frame in pixels], metrics



def summarize_latents(latents: Any, step: int) -> dict:
    """Record latent numerical health at an inference step.

    Intended to run at a few selected steps, not on every denoising step.
    Copies only scalar summary data back to the CPU.
    """
    import torch

    tensor = latents.detach()
    if tensor.numel() == 0:
        raise ValueError("Cannot inspect an empty latent tensor")

    finite = torch.isfinite(tensor)
    fraction = 1.0 - finite.count_nonzero().item() / tensor.numel()
    finite_values = tensor[finite]
    if finite_values.numel():
        minimum = float(finite_values.min().float().item())
        maximum = float(finite_values.max().float().item())
        mean_abs = float(finite_values.float().abs().mean().item())
    else:
        minimum = None
        maximum = None
        mean_abs = None

    return {
        "step": int(step),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "nonfinite_fraction": round(float(fraction), 8),
        "finite_min": minimum,
        "finite_max": maximum,
        "finite_mean_abs": mean_abs,
    }


def classify_numerical_stage(latent_checks: list[dict], frame_quality: dict) -> str:
    """Locate the first observed non-finite stage, without guessing its cause."""
    if any(row["nonfinite_fraction"] > 0 for row in latent_checks):
        return "nonfinite_observed_during_denoising"

    frame_nonfinite = frame_quality.get("nonfinite_fraction", 0) > 0
    if frame_nonfinite and latent_checks:
        return "latents_finite_at_sampled_steps_but_postdecode_nonfinite"

    if frame_nonfinite:
        return "postdecode_nonfinite_latent_stage_not_checked"

    return "no_nonfinite_detected_in_sampled_stages"



def summarize_decoded_tensor(tensor: Any) -> dict:
    """Check the raw VAE output before normalization or uint8 conversion."""
    import torch

    raw = tensor.detach()
    if raw.numel() == 0:
        raise ValueError("Cannot inspect an empty VAE tensor")

    finite = torch.isfinite(raw)
    nonfinite_fraction = 1.0 - finite.count_nonzero().item() / raw.numel()
    return {
        "dtype": str(raw.dtype),
        "shape": list(raw.shape),
        "nonfinite_fraction": round(float(nonfinite_fraction), 8),
    }


def identify_decode_stage(
    latent_checks: list[dict], raw_vae: dict, output_quality: dict
) -> str:
    if any(row["nonfinite_fraction"] > 0 for row in latent_checks):
        return "nonfinite_during_denoising"
    if raw_vae.get("nonfinite_fraction", 0) > 0:
        return "nonfinite_in_raw_vae_output"
    if output_quality.get("nonfinite_fraction", 0) > 0:
        return "nonfinite_after_vae_before_or_during_postprocess"
    return "no_nonfinite_at_observed_stages"
