"""Generate stratified train/val/test splits for the Euclid Q1 catalog.

Splits are stratified by dominant morphology class (smooth / featured /
problem / ambiguous) to ensure balanced representation across sets.

Output: data/processed/split_indices.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG = PROJECT_ROOT / "data" / "raw" / "morphology_catalogue.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "split_indices.json"


def assign_dominant_class(df: pd.DataFrame) -> pd.Series:
    """Assign each galaxy a dominant morphology class based on >50% vote."""
    conditions = [
        df["smooth-or-featured_smooth_fraction"] > 0.5,
        df["smooth-or-featured_featured-or-disk_fraction"] > 0.5,
        df["smooth-or-featured_problem_fraction"] > 0.5,
    ]
    choices = ["smooth", "featured", "problem"]
    return pd.Series(
        np.select(conditions, choices, default="ambiguous"),
        index=df.index,
    )


def make_splits(
    catalog_path: Path = DEFAULT_CATALOG,
    output_path: Path = DEFAULT_OUTPUT,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 42,
    apply_training_cuts: bool = True,
) -> dict:
    """Create stratified splits and save to JSON.

    Returns dict with 'train', 'val', 'test' keys mapping to index lists.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1, got {train_frac + val_frac + test_frac}"

    # Load and filter
    df = pd.read_parquet(catalog_path)
    if apply_training_cuts:
        df = df[~df["warning_galaxy_fails_training_cuts"]].reset_index(drop=True)

    # Stratification labels
    strat_labels = assign_dominant_class(df)

    # First split: train vs (val + test)
    val_test_frac = val_frac + test_frac
    train_idx, valtest_idx = train_test_split(
        np.arange(len(df)),
        test_size=val_test_frac,
        stratify=strat_labels,
        random_state=seed,
    )

    # Second split: val vs test (relative fraction within the held-out set)
    test_relative = test_frac / val_test_frac
    strat_valtest = strat_labels.iloc[valtest_idx]
    val_idx, test_idx = train_test_split(
        valtest_idx,
        test_size=test_relative,
        stratify=strat_valtest,
        random_state=seed,
    )

    # Build output
    splits = {
        "train": sorted(train_idx.tolist()),
        "val": sorted(val_idx.tolist()),
        "test": sorted(test_idx.tolist()),
    }

    # Validation
    all_idx = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_idx) == len(df), "Index mismatch"
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == len(df)

    # Add metadata
    meta = {
        "seed": seed,
        "total_galaxies": len(df),
        "apply_training_cuts": apply_training_cuts,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "stratification": strat_labels.value_counts().to_dict(),
        "per_split_stratification": {},
    }
    for split_name, idx_list in splits.items():
        counts = strat_labels.iloc[idx_list].value_counts().to_dict()
        meta["per_split_stratification"][split_name] = counts

    output = {"meta": meta, **splits}

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Split saved to: {output_path}")
    print(f"\nTotal galaxies: {len(df):,}")
    print(f"  Train: {len(splits['train']):>8,} ({len(splits['train'])/len(df)*100:.1f}%)")
    print(f"  Val:   {len(splits['val']):>8,} ({len(splits['val'])/len(df)*100:.1f}%)")
    print(f"  Test:  {len(splits['test']):>8,} ({len(splits['test'])/len(df)*100:.1f}%)")
    print(f"\nStratification (overall):")
    for cls, count in sorted(meta["stratification"].items()):
        print(f"  {cls:12s}: {count:>8,} ({count/len(df)*100:.1f}%)")
    print(f"\nPer-split class distribution:")
    for split_name in ("train", "val", "test"):
        counts = meta["per_split_stratification"][split_name]
        total = sum(counts.values())
        pcts = {k: f"{v/total*100:.1f}%" for k, v in counts.items()}
        print(f"  {split_name:6s}: {pcts}")

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stratified train/val/test splits")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-training-cuts", action="store_true",
                        help="Include galaxies that fail Zoobot training cuts")
    args = parser.parse_args()

    make_splits(
        catalog_path=args.catalog,
        output_path=args.output,
        seed=args.seed,
        apply_training_cuts=not args.no_training_cuts,
    )
