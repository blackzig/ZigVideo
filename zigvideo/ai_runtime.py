from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
import json
from typing import Optional


@dataclass(frozen=True)
class AIRuntimeReport:
    ready: bool
    torch_version: Optional[str]
    diffusers_version: Optional[str]
    transformers_version: Optional[str]
    accelerate_version: Optional[str]
    huggingface_hub_version: Optional[str]
    gguf_version: Optional[str]
    sentencepiece_version: Optional[str]
    protobuf_version: Optional[str]
    tiktoken_version: Optional[str]
    ltx_pipeline_available: bool
    ltx_i2v_pipeline_available: bool
    gguf_loader_available: bool
    errors: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _version(package: str) -> Optional[str]:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def inspect_ai_runtime() -> AIRuntimeReport:
    errors: list[str] = []

    torch_version = _version("torch")
    diffusers_version = _version("diffusers")
    transformers_version = _version("transformers")
    accelerate_version = _version("accelerate")
    huggingface_hub_version = _version("huggingface-hub")
    gguf_version = _version("gguf")
    sentencepiece_version = _version("sentencepiece")
    protobuf_version = _version("protobuf")
    tiktoken_version = _version("tiktoken")

    required = {
        "torch": torch_version,
        "diffusers": diffusers_version,
        "transformers": transformers_version,
        "accelerate": accelerate_version,
        "huggingface-hub": huggingface_hub_version,
        "gguf": gguf_version,
        "sentencepiece": sentencepiece_version,
        "protobuf": protobuf_version,
        "tiktoken": tiktoken_version,
    }
    for package, version in required.items():
        if version is None:
            errors.append(f"Missing package: {package}")

    ltx_pipeline_available = False
    ltx_i2v_pipeline_available = False
    gguf_loader_available = False

    if diffusers_version is not None:
        try:
            from diffusers import LTXPipeline  # noqa: F401
            ltx_pipeline_available = True
        except Exception as exc:
            errors.append(f"LTXPipeline unavailable: {type(exc).__name__}: {exc}")

        try:
            from diffusers import LTXImageToVideoPipeline  # noqa: F401
            ltx_i2v_pipeline_available = True
        except Exception as exc:
            errors.append(
                f"LTXImageToVideoPipeline unavailable: {type(exc).__name__}: {exc}"
            )

        try:
            from diffusers import GGUFQuantizationConfig, LTXVideoTransformer3DModel  # noqa: F401

            if not hasattr(LTXVideoTransformer3DModel, "from_single_file"):
                raise AttributeError(
                    "LTXVideoTransformer3DModel.from_single_file is missing"
                )
            gguf_loader_available = True
        except Exception as exc:
            errors.append(
                f"Diffusers LTX GGUF loader unavailable: {type(exc).__name__}: {exc}"
            )

    ready = (
        not errors
        and ltx_pipeline_available
        and ltx_i2v_pipeline_available
        and gguf_loader_available
    )

    return AIRuntimeReport(
        ready=ready,
        torch_version=torch_version,
        diffusers_version=diffusers_version,
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        huggingface_hub_version=huggingface_hub_version,
        gguf_version=gguf_version,
        sentencepiece_version=sentencepiece_version,
        protobuf_version=protobuf_version,
        tiktoken_version=tiktoken_version,
        ltx_pipeline_available=ltx_pipeline_available,
        ltx_i2v_pipeline_available=ltx_i2v_pipeline_available,
        gguf_loader_available=gguf_loader_available,
        errors=tuple(errors),
    )
