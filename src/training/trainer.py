"""Generic training loop for galaxy morphology models.

Supports two-phase fine-tuning (linear probe → full fine-tune),
cosine scheduling with warmup, early stopping, and checkpoint saving.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    lr: float = 5e-5
    lr_linear_probe: float = 1e-3
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 30
    linear_probe_epochs: int = 5
    full_finetune_epochs: int = 25
    warmup_fraction: float = 0.05
    patience: int = 5
    monitor: str = "val_loss"
    checkpoint_dir: str = "results/checkpoints"
    seed: int = 42


class Trainer:
    """Two-phase training loop: linear probe → full fine-tune.

    Parameters
    ----------
    model : nn.Module
        Model with a ``freeze_backbone()`` and ``unfreeze_backbone()`` method.
        If missing, only full fine-tuning is performed.
    criterion : nn.Module
        Loss function accepting (predictions, targets, masks).
    train_loader, val_loader : DataLoader
    config : TrainConfig
    device : torch.device
    model_name : str
        Used for checkpoint filenames and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
        model_name: str = "model",
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        self.model.to(self.device)

        # State
        self.history: list[dict[str, float]] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self) -> dict[str, Any]:
        """Run the full two-phase training pipeline.

        Returns a summary dict with best metrics and checkpoint path.
        """
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: Linear probe (frozen backbone)
        has_freeze = hasattr(self.model, "freeze_backbone")
        if has_freeze and self.config.linear_probe_epochs > 0:
            print(f"\n{'='*60}")
            print(f"Phase 1: Linear Probe ({self.config.linear_probe_epochs} epochs)")
            print(f"{'='*60}")
            self.model.freeze_backbone()
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.lr_linear_probe,
                weight_decay=self.config.weight_decay,
            )
            scheduler = self._make_scheduler(
                optimizer, self.config.linear_probe_epochs
            )
            self._train_epochs(
                optimizer, scheduler, self.config.linear_probe_epochs, phase="probe"
            )

        # Phase 2: Full fine-tuning
        total_ft_epochs = (
            self.config.full_finetune_epochs
            if has_freeze
            else self.config.epochs
        )
        print(f"\n{'='*60}")
        print(f"Phase 2: Full Fine-tuning ({total_ft_epochs} epochs)")
        print(f"{'='*60}")

        if has_freeze:
            self.model.unfreeze_backbone()

        # Reset early stopping for phase 2
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer, total_ft_epochs)
        self._train_epochs(
            optimizer, scheduler, total_ft_epochs, phase="finetune"
        )

        # Save final summary
        best_epoch = min(self.history, key=lambda x: x.get("val_loss", float("inf")))
        summary = {
            "model_name": self.model_name,
            "best_val_loss": best_epoch.get("val_loss"),
            "best_epoch": best_epoch.get("epoch"),
            "total_epochs_trained": len(self.history),
            "checkpoint_path": str(ckpt_dir / f"{self.model_name}_best.pt"),
            "history": self.history,
        }

        summary_path = ckpt_dir / f"{self.model_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining summary saved to: {summary_path}")

        return summary

    def _train_epochs(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        num_epochs: int,
        phase: str = "",
    ) -> None:
        """Inner training loop for a single phase."""
        for epoch_idx in range(num_epochs):
            epoch_num = len(self.history) + 1
            t0 = time.time()

            # Train
            train_loss = self._train_one_epoch(optimizer)

            # Validate
            val_loss = self._validate()

            # Step scheduler
            if scheduler is not None:
                scheduler.step()

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]

            record = {
                "epoch": epoch_num,
                "phase": phase,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "time_s": elapsed,
            }
            self.history.append(record)

            print(
                f"  Epoch {epoch_num:3d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={lr:.2e} | "
                f"{elapsed:.1f}s"
            )

            # Checkpointing + early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                ckpt_path = (
                    Path(self.config.checkpoint_dir)
                    / f"{self.model_name}_best.pt"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"    -> New best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"    -> Early stopping (patience={self.config.patience})")
                    break

    def _train_one_epoch(self, optimizer: torch.optim.Optimizer) -> float:
        """Single training epoch. Returns mean loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for images, targets, masks in tqdm(
            self.train_loader, desc="  Train", leave=False
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        """Validation pass. Returns mean loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for images, targets, masks in tqdm(
            self.val_loader, desc="  Val  ", leave=False
        ):
            images = images.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets, masks)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _make_scheduler(
        self, optimizer: torch.optim.Optimizer, num_epochs: int
    ) -> Any:
        """Cosine annealing with linear warmup."""
        total_steps = num_epochs
        warmup_steps = max(1, int(total_steps * self.config.warmup_fraction))
        cosine_steps = total_steps - warmup_steps

        warmup = LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(optimizer, T_max=max(cosine_steps, 1))

        return SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
