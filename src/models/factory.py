"""Model factory — create any model by name.

Usage:
    model = create_model("dinov2", num_outputs=22)
    model = create_model("zoobot", num_outputs=22, zoobot_checkpoint="path/to/ckpt")
"""

from __future__ import annotations

from src.models.base import MorphologyModel
from src.models.architectures import (
    ConvNeXtBase,
    DINOv2,
    SwinV2Base,
    ViTBase,
    ZoobotEfficientNet,
)

MODEL_REGISTRY: dict[str, type[MorphologyModel]] = {
    "vit-base": ViTBase,
    "swin-v2": SwinV2Base,
    "dinov2": DINOv2,
    "convnext": ConvNeXtBase,
    "zoobot": ZoobotEfficientNet,
}


def create_model(name: str, **kwargs) -> MorphologyModel:
    """Instantiate a model by its registry name.

    Parameters
    ----------
    name : one of 'vit-base', 'swin-v2', 'dinov2', 'convnext', 'zoobot'
    **kwargs : passed to the model constructor (e.g., num_outputs, pretrained)

    Returns
    -------
    MorphologyModel instance
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(MODEL_REGISTRY.keys())
