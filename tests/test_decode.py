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



def test_decode_vae_cpu_keeps_model_on_cpu():
    import torch

    from zigvideo.decode import configure_decode_vae

    class FakeVAE:
        def __init__(self):
            self.tiling = False
            self.slicing = False
            self.group_settings = None

        def enable_tiling(self):
            self.tiling = True

        def enable_slicing(self):
            self.slicing = True

        def enable_group_offload(self, **kwargs):
            self.group_settings = kwargs

    vae = FakeVAE()
    configure_decode_vae(vae, "cpu", torch)
    assert vae.tiling
    assert vae.slicing
    assert vae.group_settings is None


def test_decode_vae_cuda_uses_leaf_level_offload():
    import torch

    from zigvideo.decode import configure_decode_vae

    class FakeVAE:
        def enable_tiling(self):
            pass

        def enable_slicing(self):
            pass

        def enable_group_offload(self, **kwargs):
            self.config = kwargs

    vae = FakeVAE()
    configure_decode_vae(vae, "cuda", torch)
    assert vae.config["offload_type"] == "leaf_level"
    assert vae.config["onload_device"] == torch.device("cuda")
    assert vae.config["offload_device"] == torch.device("cpu")
    assert vae.config["use_stream"] is False


def test_decode_cli_keeps_cpu_default_and_supports_cuda():
    from zigvideo.cli import build_parser

    parser = build_parser()
    base = ["decode", "--latents", "x.safetensors", "--output", "out.mp4"]
    assert parser.parse_args(base).device == "cpu"
    assert parser.parse_args(base + ["--device", "cuda"]).device == "cuda"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decode_dry_run_checks_saved_file_without_loading_models(tmp_path, capsys, device):
    import json
    from zigvideo.cli import build_parser

    source = tmp_path / "saved.safetensors"
    save_file(
        {"latents": torch.zeros((1, 4, 16, 60, 90), dtype=torch.float16)},
        str(source),
        metadata={"width": "720", "height": "480"},
    )
    args = build_parser().parse_args(
        [
            "decode", "--latents", str(source),
            "--output", str(tmp_path / "test.mp4"),
            "--device", device, "--dry-run",
        ]
    )
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["device"] == device
    assert report["vae_precision"] == "float32"
    assert report["offload_strategy"] == (
        "leaf_level" if device == "cuda" else "none"
    )



def test_gpu_allocator_peak_flag_exceeds_device_capacity():
    from zigvideo.decode import peak_allocation_exceeds_device_memory

    assert peak_allocation_exceeds_device_memory(10.574, 6.0) is True
    assert peak_allocation_exceeds_device_memory(3.8, 6.0) is False


def test_gpu_allocator_peak_flag_cpu_and_invalid_capacity():
    from zigvideo.decode import peak_allocation_exceeds_device_memory

    assert peak_allocation_exceeds_device_memory(None, None) is None
    assert peak_allocation_exceeds_device_memory(3.8, None) is None
    with pytest.raises(ValueError, match="Invalid GPU"):
        peak_allocation_exceeds_device_memory(1.0, 0.0)
