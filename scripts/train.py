"""CLI training entry point for galaxy morphology models.

Usage:
    python scripts/train.py --model zoobot                        # Zoobot baseline
    python scripts/train.py --model dinov2 --batch-size 64        # DINOv2 on A100
    python scripts/train.py --model vit-base --loss mse           # ViT with MSE loss
    python scripts/train.py --model dinov2 --epochs 10 --lr 1e-4  # Quick test
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import EuclidDataset, MorphologySchema
from src.data.transforms import get_transforms
from src.models.factory import create_model, list_models
from src.training.losses import DirichletMultinomialLoss, MaskedMSELoss
from src.training.trainer import Trainer, TrainConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train a galaxy morphology model")
    parser.add_argument("--model", type=str, required=True, choices=list_models(),
                        help=f"Model name: {list_models()}")
    parser.add_argument("--config", type=str, default="configs/base.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--loss", type=str, default=None, choices=["dirichlet_multinomial", "mse"],
                        help="Override loss function from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total epochs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch (no pretrained weights)")
    parser.add_argument("--zoobot-checkpoint", type=str, default=None,
                        help="Path to Zoobot pretrained weights (for zoobot model)")
    args = parser.parse_args()

    # Load config
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply overrides
    seed = args.seed or cfg["seed"]
    seed_everything(seed)

    batch_size = args.batch_size or cfg["training"]["batch_size"]
    lr = args.lr or cfg["training"]["lr"]
    loss_name = args.loss or cfg["training"]["loss"]
    total_epochs = args.epochs or cfg["training"]["epochs"]

    # Schema
    schema = MorphologySchema.from_yaml(cfg["data"]["morphology_questions"])
    print(f"Schema: {schema.num_outputs} outputs across {len(schema.questions)} questions")

    # Data
    catalog_path = project_root / cfg["data"]["catalog_path"]
    split_path = project_root / cfg["data"]["split_path"]
    image_dir = project_root / cfg["data"]["image_dir"]
    input_size = cfg["data"]["input_size"]

    print(f"\nLoading data splits from: {split_path}")
    train_df = EuclidDataset.load_split(catalog_path, split_path, "train")
    val_df = EuclidDataset.load_split(catalog_path, split_path, "val")

    train_transform = get_transforms("train", input_size, augmentation_cfg=cfg["augmentation"]["train"])
    val_transform = get_transforms("val", input_size)

    train_ds = EuclidDataset(train_df, image_dir, schema, train_transform, cfg["data"]["image_type"])
    val_ds = EuclidDataset(val_df, image_dir, schema, val_transform, cfg["data"]["image_type"])

    num_workers = cfg["data"].get("num_workers", 4)
    pin_memory = cfg["data"].get("pin_memory", True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    print(f"Train: {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds):,} samples ({len(val_loader)} batches)")

    # Model
    model_kwargs = {
        "num_outputs": schema.num_outputs,
        "pretrained": not args.no_pretrained,
    }
    if args.model == "zoobot" and args.zoobot_checkpoint:
        model_kwargs["zoobot_checkpoint"] = args.zoobot_checkpoint

    model = create_model(args.model, **model_kwargs)
    params = model.count_parameters()
    print(f"\nModel: {args.model}")
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Loss
    if loss_name == "dirichlet_multinomial":
        criterion = DirichletMultinomialLoss(schema)
    else:
        criterion = MaskedMSELoss()
    print(f"Loss: {loss_name}")

    # Trainer config
    train_config = TrainConfig(
        lr=lr,
        lr_linear_probe=cfg["training"]["lr_linear_probe"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=batch_size,
        epochs=total_epochs,
        linear_probe_epochs=cfg["training"]["finetune"]["linear_probe_epochs"],
        full_finetune_epochs=cfg["training"]["finetune"]["full_finetune_epochs"],
        warmup_fraction=cfg["training"]["scheduler"]["warmup_fraction"],
        patience=cfg["training"]["early_stopping"]["patience"],
        checkpoint_dir=str(project_root / "results" / "checkpoints"),
        seed=seed,
    )

    # Override epochs if specified
    if args.epochs:
        train_config.linear_probe_epochs = min(5, args.epochs // 3)
        train_config.full_finetune_epochs = args.epochs - train_config.linear_probe_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Train
    trainer = Trainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
        model_name=args.model,
    )

    summary = trainer.train()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE — {args.model}")
    print(f"{'='*60}")
    print(f"Best val_loss: {summary['best_val_loss']:.4f} (epoch {summary['best_epoch']})")
    print(f"Checkpoint: {summary['checkpoint_path']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
