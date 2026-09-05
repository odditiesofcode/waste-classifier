"""
Data loading and splitting for the waste classifier.

Design decision: this module produces a *manifest* (a CSV mapping each
image path to a class and a split) rather than immediately loading pixel
data into a PyTorch/TF Dataset. Why: it keeps the "which image goes in
which split" logic independent of whatever training framework we pick,
and independently testable (see tests/test_data.py) without needing
torch/tensorflow installed at all. The framework-specific Dataset class
(step 2) will just read this manifest.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent.parent / "data" / "dataset-resized"
SPLITS_PATH = Path(__file__).parent.parent / "data" / "splits.csv"

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def build_manifest(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Walk the class folders and build a flat (path, class) table."""
    rows = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir()):
            if img_path.name.startswith("."):
                continue
            rows.append({"path": str(img_path), "class": class_dir.name})
    return pd.DataFrame(rows)


def stratified_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Split into train/val/test, preserving each class's proportion in
    every split (stratification).

    Why stratify: with plain random splitting, a small class like
    `trash` (137 images) could easily end up under- or over-represented
    in the test set purely by chance, making our eval numbers noisy and
    hard to trust run-to-run. Stratifying removes that source of noise.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        "split fractions must sum to 1"

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["class"],
        random_state=seed,
    )
    # split the remainder proportionally into val/test
    relative_val_frac = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_frac,
        stratify=temp_df["class"],
        random_state=seed,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def compute_class_weights(df: pd.DataFrame, split: str = "train") -> dict:
    """
    Inverse-frequency class weights, computed from the TRAINING split only.

    Why train-only: val/test are meant to represent the real-world
    distribution we'll be evaluated against. If we computed weights from
    the full dataset (or leaked val/test info into this calculation),
    we'd be letting evaluation data influence training decisions —
    a subtle form of data leakage.

    These weights get passed to the loss function later so the model is
    penalized more for getting `trash` wrong than for getting `paper`
    wrong, counteracting the 4.3x imbalance we saw in EDA.
    """
    train_counts = df[df["split"] == split]["class"].value_counts()
    n_classes = len(train_counts)
    n_samples = train_counts.sum()
    # standard inverse-frequency weighting, normalized so weights average to 1
    weights = n_samples / (n_classes * train_counts)
    return weights.to_dict()


def build_and_save_splits() -> pd.DataFrame:
    manifest = build_manifest()
    split_df = stratified_split(manifest)
    split_df.to_csv(SPLITS_PATH, index=False)
    return split_df


if __name__ == "__main__":
    split_df = build_and_save_splits()

    print("Split sizes:")
    print(split_df["split"].value_counts().to_string())

    print("\nClass balance within each split (should look similar across rows):")
    print(pd.crosstab(split_df["class"], split_df["split"], normalize="columns").round(3).to_string())

    weights = compute_class_weights(split_df)
    print("\nClass weights (for the loss function):")
    for cls, w in sorted(weights.items()):
        print(f"  {cls:<10} {w:.3f}")

    print(f"\nSaved manifest+splits to {SPLITS_PATH}")
