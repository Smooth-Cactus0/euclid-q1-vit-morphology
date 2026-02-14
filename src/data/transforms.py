"""Image transforms for Euclid Q1 galaxy morphology.

Provides separate transform pipelines for training, validation/test,
and test-time augmentation (TTA).
"""

from __future__ import annotations

from torchvision import transforms


def get_transforms(
    split: str = "train",
    input_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    augmentation_cfg: dict | None = None,
) -> transforms.Compose:
    """Build a torchvision transform pipeline.

    Parameters
    ----------
    split : 'train', 'val', 'test', or 'tta'
    input_size : target spatial size (square)
    mean, std : per-channel normalization (default: ImageNet)
    augmentation_cfg : optional dict from configs/base.yaml ``augmentation.train``
    """
    if split == "train":
        return _train_transforms(input_size, mean, std, augmentation_cfg)
    elif split in ("val", "test"):
        return _eval_transforms(input_size, mean, std)
    else:
        raise ValueError(f"Unknown split: {split!r}. Use 'train', 'val', or 'test'.")


def get_tta_transforms(
    input_size: int = 224,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> list[transforms.Compose]:
    """Return a list of 7 deterministic transforms for TTA.

    Each produces a different augmented view of the same image.
    Average the model's predictions across all views.
    """
    base = [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
    ]
    norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    tta_list = [
        # Original
        transforms.Compose(base + norm),
        # Horizontal flip
        transforms.Compose(base + [transforms.RandomHorizontalFlip(p=1.0)] + norm),
        # Vertical flip
        transforms.Compose(base + [transforms.RandomVerticalFlip(p=1.0)] + norm),
        # 90° rotation
        transforms.Compose(base + [transforms.Lambda(lambda img: img.rotate(90))] + norm),
        # 180° rotation
        transforms.Compose(base + [transforms.Lambda(lambda img: img.rotate(180))] + norm),
        # 270° rotation
        transforms.Compose(base + [transforms.Lambda(lambda img: img.rotate(270))] + norm),
        # Horizontal flip + 90° rotation
        transforms.Compose(
            base
            + [transforms.RandomHorizontalFlip(p=1.0),
               transforms.Lambda(lambda img: img.rotate(90))]
            + norm
        ),
    ]
    return tta_list


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _train_transforms(
    input_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
    cfg: dict | None,
) -> transforms.Compose:
    """Training pipeline: augmentation + normalization."""
    cfg = cfg or {}
    ops = []

    # Resize to slightly larger than input, then random crop
    crop_cfg = cfg.get("random_resized_crop", {})
    scale = tuple(crop_cfg.get("scale", [0.85, 1.0]))
    ratio = tuple(crop_cfg.get("ratio", [0.95, 1.05]))
    ops.append(transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio))

    # Flips — galaxies have no preferred orientation
    if cfg.get("random_horizontal_flip", True):
        ops.append(transforms.RandomHorizontalFlip())
    if cfg.get("random_vertical_flip", True):
        ops.append(transforms.RandomVerticalFlip())

    # Rotation — full 360° for galaxies
    rotation = cfg.get("random_rotation", 360)
    if rotation > 0:
        ops.append(transforms.RandomRotation(rotation))

    # Color jitter
    jitter_cfg = cfg.get("color_jitter", {})
    if any(jitter_cfg.get(k, 0) > 0 for k in ("brightness", "contrast", "saturation", "hue")):
        ops.append(transforms.ColorJitter(
            brightness=jitter_cfg.get("brightness", 0),
            contrast=jitter_cfg.get("contrast", 0),
            saturation=jitter_cfg.get("saturation", 0),
            hue=jitter_cfg.get("hue", 0),
        ))

    # To tensor + normalize
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


def _eval_transforms(
    input_size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> transforms.Compose:
    """Eval pipeline: deterministic resize + center crop + normalization."""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
