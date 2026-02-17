"""Euclid Q1 Galaxy Morphology Dataset.

Provides a PyTorch Dataset that loads galaxy cutout images and returns
vote-fraction targets with a validity mask for tree-structured questions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Morphology schema — maps question names to column suffixes
# ---------------------------------------------------------------------------

@dataclass
class MorphologySchema:
    """Describes the Galaxy Zoo decision-tree questions we regress on.

    Each question has a list of answer names.  The corresponding catalog
    columns are ``{question}_{answer}_fraction``.  The answers within each
    question always sum to 1.

    The schema also tracks which output indices correspond to each question,
    so the training loop can build per-question masked losses.
    """

    questions: dict[str, list[str]] = field(default_factory=dict)

    # Computed: maps question -> (start_idx, end_idx) in the flat target tensor
    question_slices: dict[str, tuple[int, int]] = field(
        default_factory=dict, init=False, repr=False,
    )
    # Computed: ordered list of all column names
    columns: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_index()

    def _build_index(self) -> None:
        self.columns = []
        self.question_slices = {}
        idx = 0
        for question, answers in self.questions.items():
            start = idx
            for answer in answers:
                self.columns.append(f"{question}_{answer}_fraction")
                idx += 1
            self.question_slices[question] = (start, idx)

    @property
    def num_outputs(self) -> int:
        return len(self.columns)

    @classmethod
    def from_yaml(cls, cfg: dict) -> MorphologySchema:
        """Build schema from the ``data.morphology_questions`` section of a
        YAML config file."""
        questions = {
            name: spec["answers"]
            for name, spec in cfg.items()
        }
        return cls(questions=questions)

    @classmethod
    def default(cls) -> MorphologySchema:
        """The 7 main Galaxy Zoo Euclid questions (22 outputs)."""
        return cls(questions={
            "smooth-or-featured": ["smooth", "featured-or-disk", "problem"],
            "disk-edge-on": ["yes", "no"],
            "has-spiral-arms": ["yes", "no"],
            "bar": ["strong", "weak", "no"],
            "bulge-size": ["dominant", "large", "moderate", "small", "none"],
            "how-rounded": ["round", "in-between", "cigar-shaped"],
            "merging": ["none", "minor-disturbance", "major-disturbance", "merger"],
        })


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EuclidDataset(Dataset):
    """PyTorch Dataset for Euclid Q1 galaxy morphology.

    Parameters
    ----------
    catalog : pd.DataFrame
        Subset of the morphology catalog (already filtered and split).
    image_dir : str | Path
        Directory containing galaxy cutout JPGs, organised as
        ``<tile_index>/<tile>_<object_id>_<type>.jpg``.
    schema : MorphologySchema
        Defines which questions/answers to regress on.
    transform : callable, optional
        Torchvision-style transform applied to PIL images.
    image_type : str
        Suffix used in the JPG filenames (default: ``gz_arcsinh_vis_y``).

    Returns
    -------
    image : Tensor [C, H, W]
    targets : Tensor [num_outputs]
        Vote fractions.  NaN questions are filled with 0.
    mask : Tensor [num_outputs]
        1.0 where the vote fraction is valid, 0.0 where it was NaN.
        Use this to mask the loss: ``loss = (loss_per_output * mask).sum()``.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        image_dir: str | Path,
        schema: MorphologySchema | None = None,
        transform: Callable | None = None,
        image_type: str = "gz_arcsinh_vis_y",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.schema = schema or MorphologySchema.default()
        self.transform = transform
        self.image_type = image_type

        # Pre-extract the columns we need as numpy arrays for speed
        self.catalog = catalog.reset_index(drop=True)
        self.tile_indices = self.catalog["tile_index"].values
        self.object_ids = self.catalog["object_id"].values

        # Build target matrix [N, num_outputs] and mask matrix
        target_df = self.catalog[self.schema.columns]
        self.targets = torch.from_numpy(target_df.values.astype(np.float32))
        self.masks = (~torch.isnan(self.targets)).float()
        # Replace NaN with 0 so tensors are safe for computation
        self.targets = torch.nan_to_num(self.targets, nan=0.0)

    def __len__(self) -> int:
        return len(self.catalog)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build image path: <tile>/<tile>_<obj_id>_<type>.jpg
        tile = self.tile_indices[idx]
        obj_id = str(self.object_ids[idx]).replace("-", "NEG")
        fname = f"{tile}_{obj_id}_{self.image_type}.jpg"
        img_path = self.image_dir / str(tile) / fname

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, self.targets[idx], self.masks[idx]

    def get_question_mask(self, question: str) -> torch.Tensor:
        """Return a boolean mask [N] for galaxies that have valid votes
        for the given question."""
        start, end = self.schema.question_slices[question]
        # A question is valid if *any* of its answer columns are non-NaN
        return self.masks[:, start] > 0

    @staticmethod
    def load_split(
        catalog_path: str | Path,
        split_path: str | Path,
        split: str,
        apply_training_cuts: bool = True,
    ) -> pd.DataFrame:
        """Load a catalog subset for a given split (train/val/test).

        Parameters
        ----------
        catalog_path : path to the parquet catalog
        split_path : path to split_indices.json
        split : one of 'train', 'val', 'test'
        apply_training_cuts : if True, filter out galaxies that fail Zoobot cuts
        """
        catalog = pd.read_parquet(catalog_path)
        if apply_training_cuts:
            catalog = catalog[~catalog["warning_galaxy_fails_training_cuts"]]

        with open(split_path) as f:
            split_indices = json.load(f)

        indices = split_indices[split]
        return catalog.iloc[indices].reset_index(drop=True)
