"""Base model interface for galaxy morphology regression.

All models implement this interface so the Trainer, evaluation, and
interpretability code can work with any architecture uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class MorphologyModel(ABC, nn.Module):
    """Abstract base class for galaxy morphology models.

    Subclasses must implement:
    - ``forward(x)`` → raw logits of shape [B, num_outputs]
    - ``freeze_backbone()`` / ``unfreeze_backbone()`` for two-phase fine-tuning
    """

    def __init__(self, num_outputs: int = 22) -> None:
        super().__init__()
        self.num_outputs = num_outputs

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : [B, 3, H, W] input images

        Returns
        -------
        [B, num_outputs] raw logits (not softmaxed — loss handles activation)
        """

    @abstractmethod
    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for linear probe phase)."""

    @abstractmethod
    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for full fine-tune phase)."""

    def count_parameters(self) -> dict[str, int]:
        """Return total, trainable, and frozen parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
