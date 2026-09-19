"""
Evaluation script — run this AFTER training, on the test set (never
touched until now, not even for the "which checkpoint is best" decision
made during training, which used val).

Why test, not val, for the final report: val accuracy was used to pick
which checkpoint to keep, which means val has already influenced our
decisions once. Test is the one number/report we haven't looked at or
optimized against at all, so it's the most honest estimate of real
performance.

Usage:
    python -m src.evaluate
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from src.dataset import WasteDataset, CLASSES
from src.model import build_model

CHECKPOINT_PATH = Path(__file__).parent.parent / "checkpoints" / "best_model.pt"
OUTPUT_DIR = Path(__file__).parent.parent / "checkpoints"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    test_ds = WasteDataset("test")
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    all_preds, all_labels, all_confidences, all_paths = [], [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confidences.extend(confidences.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_paths = test_ds.df["path"].values  # same order as the DataLoader, since shuffle=False

    # --- Overall + per-class metrics ---
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS (test set)")
    print("=" * 60)
    report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=3)
    print(report)
    with open(OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")

    # annotate each cell with its count, so the plot is readable without
    # needing a separate legend
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=130)
    print(f"\nSaved confusion matrix to {OUTPUT_DIR / 'confusion_matrix.png'}")

    # --- Error analysis: which specific images got misclassified, and how confidently ---
    misclassified_mask = all_preds != all_labels
    error_df = pd.DataFrame({
        "path": all_paths[misclassified_mask],
        "true_class": [CLASSES[i] for i in all_labels[misclassified_mask]],
        "predicted_class": [CLASSES[i] for i in all_preds[misclassified_mask]],
        "confidence": np.array(all_confidences)[misclassified_mask],
    }).sort_values("confidence", ascending=False)

    error_df.to_csv(OUTPUT_DIR / "misclassified_images.csv", index=False)
    print(f"\n{len(error_df)} / {len(all_labels)} test images misclassified "
          f"({100 * len(error_df) / len(all_labels):.1f}%)")
    print(f"Full list saved to {OUTPUT_DIR / 'misclassified_images.csv'}")

    print("\nMost confidently WRONG predictions (worth looking at first —")
    print("these are cases the model is confused about in a big way, not")
    print("just borderline):")
    print(error_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
