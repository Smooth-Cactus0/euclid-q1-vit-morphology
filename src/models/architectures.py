"""Concrete model implementations for galaxy morphology regression.

Each model wraps a pretrained backbone from timm or transformers,
replaces the classification head with a regression head outputting
``num_outputs`` logits, and implements freeze/unfreeze for two-phase training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm

from src.models.base import MorphologyModel


# ---------------------------------------------------------------------------
# ViT-Base/16 (vanilla Vision Transformer)
# ---------------------------------------------------------------------------

class ViTBase(MorphologyModel):
    """ViT-Base/16 pretrained on ImageNet-21k."""

    def __init__(self, num_outputs: int = 22, pretrained: bool = True) -> None:
        super().__init__(num_outputs)
        self.backbone = timm.create_model(
            "vit_base_patch16_224", pretrained=pretrained, num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Swin-V2-Base (hierarchical vision transformer)
# ---------------------------------------------------------------------------

class SwinV2Base(MorphologyModel):
    """Swin Transformer V2 Base pretrained on ImageNet-21k."""

    def __init__(self, num_outputs: int = 22, pretrained: bool = True) -> None:
        super().__init__(num_outputs)
        self.backbone = timm.create_model(
            "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# DINOv2 ViT-B/14 (self-supervised)
# ---------------------------------------------------------------------------

class DINOv2(MorphologyModel):
    """DINOv2 ViT-B/14 with frozen or fine-tuned backbone."""

    def __init__(self, num_outputs: int = 22, pretrained: bool = True) -> None:
        super().__init__(num_outputs)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14",
        ) if pretrained else timm.create_model(
            "vit_base_patch14_dinov2", pretrained=False, num_classes=0,
        )
        # DINOv2 outputs 768-dim features
        self.head = nn.Linear(768, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DINOv2's forward returns CLS token features
        features = self.backbone(x)
        # Handle different return types (some versions return dict)
        if isinstance(features, dict):
            features = features["x_norm_clstok"]
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# ConvNeXt-Base (modern CNN baseline)
# ---------------------------------------------------------------------------

class ConvNeXtBase(MorphologyModel):
    """ConvNeXt-Base pretrained on ImageNet-21k."""

    def __init__(self, num_outputs: int = 22, pretrained: bool = True) -> None:
        super().__init__(num_outputs)
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=pretrained,
            num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Zoobot (EfficientNet-B0 — the published baseline)
# ---------------------------------------------------------------------------

class ZoobotEfficientNet(MorphologyModel):
    """EfficientNet-B0 as used by Zoobot.

    This can load either:
    - Standard ImageNet-pretrained EfficientNet-B0 (default)
    - Zoobot's Galaxy Zoo pretrained weights (via ``from_zoobot=True``)

    For Zoobot pretrained weights, download from:
    https://github.com/mwalmsley/zoobot
    and pass the checkpoint path.
    """

    def __init__(
        self,
        num_outputs: int = 22,
        pretrained: bool = True,
        zoobot_checkpoint: str | None = None,
    ) -> None:
        super().__init__(num_outputs)
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0,
        )
        self.head = nn.Linear(self.backbone.num_features, num_outputs)

        if zoobot_checkpoint is not None:
            self._load_zoobot_weights(zoobot_checkpoint)

    def _load_zoobot_weights(self, checkpoint_path: str) -> None:
        """Load pretrained Zoobot weights (backbone only)."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        # Zoobot checkpoints may have different key prefixes
        backbone_dict = {}
        for k, v in state_dict.items():
            # Strip common prefixes from Zoobot checkpoints
            for prefix in ("encoder.", "model.encoder.", "backbone."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            backbone_dict[k] = v

        missing, unexpected = self.backbone.load_state_dict(
            backbone_dict, strict=False,
        )
        if missing:
            print(f"  Zoobot load — missing keys: {len(missing)}")
        if unexpected:
            print(f"  Zoobot load — unexpected keys: {len(unexpected)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
