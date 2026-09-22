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
            rows.append({"path": str(img_path), "class": class_dir.name, "source": "trashnet"})
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
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, \
        "split fractions must sum to 1"

    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["class"],
        random_state=seed,
    )
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
    """Inverse-frequency class weights, computed from the TRAINING split only."""
    train_counts = df[df["split"] == split]["class"].value_counts()
    n_classes = len(train_counts)
    n_samples = train_counts.sum()
    weights = n_samples / (n_classes * train_counts)
    return weights.to_dict()


def build_and_save_splits() -> pd.DataFrame:
    manifest = build_manifest()
    split_df = stratified_split(manifest)
    split_df.to_csv(SPLITS_PATH, index=False)
    return split_df


# --- RealWaste integration -------------------------------------------------
#
# RealWaste (UCI ML Repo, CC BY 4.0) is real-world landfill photos across
# 9 material types. We're using it for two purposes: (1) mixed into
# training to give the model exposure to messy, real-world lighting and
# backgrounds it's never seen in TrashNet's clean studio photos, and
# (2) a held-out chunk that NEVER enters training, used purely to
# measure the studio-to-real-world generalization gap.

REALWASTE_DIR = Path(__file__).parent.parent / "data" / "RealWaste"
REAL_WORLD_HOLDOUT_PATH = Path(__file__).parent.parent / "data" / "real_world_holdout.csv"
REALWASTE_HOLDOUT_FRAC = 0.30

# RealWaste has 9 classes; we only have 6. DECISION (variant A, see
# conversation/README): Cardboard/Glass/Metal/Paper/Plastic map directly
# onto our existing classes. "Miscellaneous Trash" is close enough in
# meaning to our TrashNet-defined `trash` ("doesn't fit anywhere else")
# to merge in.
#
# Food Organics, Textile Trash, and Vegetation are deliberately DROPPED
# (mapped to None, filtered out in build_realwaste_manifest) rather than
# folded into `trash`. Why: merging them in ballooned `trash` from 5% to
# 22% of the combined dataset, which flipped which classes the
# loss-weighting treats as "rare" -- a real side effect worth avoiding
# for now, since it would muddy the before/after comparison the
# real-world holdout set exists to give us.
#
# FUTURE WORK (variant C, deferred): split these into their own
# `organic_other` class rather than dropping them -- more semantically
# honest (compostable waste is a genuinely different disposal stream
# than landfill trash) and keeps ~1,165 images we're currently
# discarding. Deferred because it changes the model from 6 classes to
# 7, which breaks direct comparison against the checkpoint we already
# have and requires retraining from scratch rather than continuing to
# build on current results.
REALWASTE_CLASS_MAP = {
    "Cardboard": "cardboard",
    "Glass": "glass",
    "Metal": "metal",
    "Paper": "paper",
    "Plastic": "plastic",
    "Miscellaneous Trash": "trash",
    "Food Organics": None,
    "Textile Trash": None,
    "Vegetation": None,
}


def build_realwaste_manifest(data_dir: Path = REALWASTE_DIR) -> pd.DataFrame:
    """Same shape as build_manifest(), but remaps RealWaste's 9 folder
    names onto our 6-class taxonomy via REALWASTE_CLASS_MAP, dropping
    any class mapped to None."""
    rows = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name not in REALWASTE_CLASS_MAP:
            continue  # unrecognized folder name -- skip defensively rather than crash
        mapped_class = REALWASTE_CLASS_MAP[class_dir.name]
        if mapped_class is None:
            continue  # deliberately dropped class (variant A)
        for img_path in sorted(class_dir.iterdir()):
            if img_path.name.startswith("."):
                continue
            rows.append({"path": str(img_path), "class": mapped_class, "source": "realwaste"})
    return pd.DataFrame(rows)


def reserve_real_world_holdout(
    realwaste_df: pd.DataFrame,
    holdout_frac: float = REALWASTE_HOLDOUT_FRAC,
    seed: int = RANDOM_SEED,
):
    """
    Split RealWaste into two non-overlapping pieces:

      holdout    -- NEVER touched by training or validation, at all.
      trainable  -- the remaining 70%, folded into train/val/test
                    alongside TrashNet.

    Stratified by class so the holdout isn't accidentally skewed toward
    one class by chance.
    """
    holdout_df, trainable_df = train_test_split(
        realwaste_df,
        train_size=holdout_frac,
        stratify=realwaste_df["class"],
        random_state=seed,
    )
    return holdout_df.reset_index(drop=True), trainable_df.reset_index(drop=True)


def build_and_save_combined_splits() -> pd.DataFrame:
    """
    Full pipeline: TrashNet + (most of) RealWaste -> train/val/test,
    with a pure RealWaste holdout saved separately.

    Falls back to TrashNet-only behavior if RealWaste hasn't been
    downloaded yet, so this stays runnable at every stage of the project.
    """
    trashnet_df = build_manifest()

    if REALWASTE_DIR.exists():
        realwaste_df = build_realwaste_manifest()
        holdout_df, trainable_df = reserve_real_world_holdout(realwaste_df)
        holdout_df.to_csv(REAL_WORLD_HOLDOUT_PATH, index=False)
        combined_df = pd.concat([trashnet_df, trainable_df], ignore_index=True)
    else:
        print(f"Note: {REALWASTE_DIR} not found -- building TrashNet-only splits. "
              f"Run scripts/download_realwaste.sh to include RealWaste.")
        combined_df = trashnet_df

    split_df = stratified_split(combined_df)
    split_df.to_csv(SPLITS_PATH, index=False)
    return split_df


if __name__ == "__main__":
    split_df = build_and_save_combined_splits()

    print("Split sizes:")
    print(split_df["split"].value_counts().to_string())

    if "source" in split_df.columns:
        print("\nSource breakdown per split:")
        print(pd.crosstab(split_df["source"], split_df["split"]).to_string())

    print("\nClass balance within each split (should look similar across rows):")
    print(pd.crosstab(split_df["class"], split_df["split"], normalize="columns").round(3).to_string())

    weights = compute_class_weights(split_df)
    print("\nClass weights (for the loss function):")
    for cls, w in sorted(weights.items()):
        print(f"  {cls:<10} {w:.3f}")

    if REAL_WORLD_HOLDOUT_PATH.exists():
        holdout_df = pd.read_csv(REAL_WORLD_HOLDOUT_PATH)
        print(f"\nReal-world holdout set: {len(holdout_df)} images, never used in training/val/test")
        print(holdout_df["class"].value_counts().to_string())

    print(f"\nSaved splits to {SPLITS_PATH}")
