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
