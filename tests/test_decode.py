import torch
import pytest
from safetensors.torch import save_file

from zigvideo.decode import inspect_latent_file
from zigvideo.quality import summarize_decoded_tensor


def test_saved_cogvideox_latents_have_expected_layout(tmp_path):
    path = tmp_path / "latents.safetensors"
    # CogVideoX stores 4 *latent* frames and 16 channels for this 16-frame clip.
    source = torch.zeros((1, 4, 16, 60, 90), dtype=torch.float16)
    save_file(
        {"latents": source},
        str(path),
        metadata={"width": "720", "height": "480", "frames": "16"},
    )
    info = inspect_latent_file(path)
    assert info["shape"] == [1, 4, 16, 60, 90]
    assert info["native_resolution"] == "720x480"
    assert info["latent_frame_count"] == 4
    assert info["metadata"]["frames"] == "16"


def test_reject_incorrect_cogvideox_latent_layout(tmp_path):
    path = tmp_path / "wrong.safetensors"
    save_file(
        {"latents": torch.zeros((1, 16, 4, 60, 90), dtype=torch.float16)},
        str(path),
    )
    with pytest.raises(ValueError, match="Expected CogVideoX latents"):
        inspect_latent_file(path)


def test_raw_vae_distribution_checks_clipping():
    raw = torch.tensor(
        [[[[[-2.0, -1.5, 0.0, 0.5, 1.0]]]]], dtype=torch.float32
    )
    stats = summarize_decoded_tensor(raw)
    assert stats["nonfinite_fraction"] == 0.0
    assert stats["sample_min"] == -2.0
    assert stats["sample_max"] == 1.0
    assert stats["sample_fraction_below_minus_one"] == 0.4
