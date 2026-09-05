"""
Exploratory Data Analysis — TrashNet waste classification dataset.

Why this comes first: before writing a single line of model code, we need
to know what we're actually working with — how many images per class,
whether they're all readable, what sizes/formats they're in, and whether
anything looks broken. Skipping this step is the #1 way people end up
debugging a "model" problem that was actually a data problem.
"""

import os
from pathlib import Path
from collections import Counter

import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).parent.parent / "data" / "dataset-resized"
OUTPUT_DIR = Path(__file__).parent
CLASSES = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])


def scan_dataset():
    """
    Walk every image in every class folder and record metadata about it.

    We don't just trust that the images are all valid, correctly sized,
    and correctly labeled — we verify it. Corrupt files or mislabeled
    folders are extremely common in "real" datasets scraped or collected
    by hand (this one was collected by two Stanford students with a
    phone camera), and they fail silently if you don't check.
    """
    records = []
    corrupt_files = []

    for class_name in CLASSES:
        class_dir = DATA_DIR / class_name
        for img_path in class_dir.iterdir():
            if img_path.name.startswith("."):
                continue
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    mode = img.mode
                    file_size_kb = img_path.stat().st_size / 1024
                records.append({
                    "class": class_name,
                    "filename": img_path.name,
                    "width": width,
                    "height": height,
                    "aspect_ratio": round(width / height, 3),
                    "mode": mode,
                    "file_size_kb": round(file_size_kb, 1),
                })
            except Exception as e:
                corrupt_files.append((str(img_path), str(e)))

    return pd.DataFrame(records), corrupt_files


def summarize(df: pd.DataFrame, corrupt_files: list):
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nTotal images found: {len(df)}")
    print(f"Corrupt/unreadable files: {len(corrupt_files)}")
    if corrupt_files:
        for path, err in corrupt_files:
            print(f"  - {path}: {err}")

    print("\n--- Class distribution ---")
    counts = df["class"].value_counts()
    total = len(df)
    for cls, count in counts.items():
        pct = 100 * count / total
        bar = "#" * int(pct)
        print(f"  {cls:<10} {count:>4}  ({pct:5.1f}%)  {bar}")

    imbalance_ratio = counts.max() / counts.min()
    print(f"\n  Largest/smallest class ratio: {imbalance_ratio:.2f}x "
          f"({counts.idxmax()}={counts.max()} vs {counts.idxmin()}={counts.min()})")
    if imbalance_ratio > 2:
        print("  -> Meaningful imbalance. We'll need to account for this in "
              "training (class weights or a weighted sampler), or the model "
              "will just learn to favor the majority classes.")

    print("\n--- Image dimensions ---")
    print(f"  Unique (width, height) pairs: {df[['width','height']].drop_duplicates().shape[0]}")
    print(df[["width", "height"]].describe().loc[["min", "mean", "max"]].to_string())

    print("\n--- Color mode ---")
    print(df["mode"].value_counts().to_string())

    print("\n--- File size (KB) ---")
    print(df["file_size_kb"].describe().to_string())

    return counts


def make_plots(df: pd.DataFrame, counts: pd.Series):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts.reindex(CLASSES).plot(kind="bar", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Images per class")
    axes[0].set_ylabel("count")
    axes[0].tick_params(axis="x", rotation=45)

    for cls in CLASSES:
        subset = df[df["class"] == cls]
        axes[1].scatter(subset["width"], subset["height"], s=8, alpha=0.4, label=cls)
    axes[1].set_title("Image dimensions by class")
    axes[1].set_xlabel("width")
    axes[1].set_ylabel("height")
    axes[1].legend(fontsize=7, markerscale=2)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_summary.png"
    plt.savefig(out_path, dpi=130)
    print(f"\nSaved plot to {out_path}")


def make_sample_grid():
    """Save a small grid of sample images per class, so we can eyeball
    quality/labeling before trusting the dataset."""
    import matplotlib.pyplot as plt

    n_samples = 4
    fig, axes = plt.subplots(len(CLASSES), n_samples, figsize=(n_samples * 2, len(CLASSES) * 2))

    for row, cls in enumerate(CLASSES):
        class_dir = DATA_DIR / cls
        files = sorted([f for f in class_dir.iterdir() if not f.name.startswith(".")])[:n_samples]
        for col, f in enumerate(files):
            img = Image.open(f)
            axes[row, col].imshow(img)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_ylabel(cls)
                axes[row, col].text(-0.15, 0.5, cls, transform=axes[row, col].transAxes,
                                     rotation=90, va="center", ha="center", fontsize=10)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "sample_grid.png"
    plt.savefig(out_path, dpi=130)
    print(f"Saved sample grid to {out_path}")


if __name__ == "__main__":
    df, corrupt_files = scan_dataset()
    counts = summarize(df, corrupt_files)
    make_plots(df, counts)
    make_sample_grid()
    df.to_csv(OUTPUT_DIR / "image_manifest.csv", index=False)
    print(f"\nFull per-image manifest saved to {OUTPUT_DIR / 'image_manifest.csv'}")
