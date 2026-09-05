"""
Training script.

Usage (after `pip install -r requirements.txt -r requirements-train.txt`
and `bash scripts/download_data.sh` and `python src/data.py` to build splits):

    python src/train.py --epochs 15 --batch-size 32

Run on Colab: clone the repo, run those same commands in a cell after
switching Runtime > Change runtime type > GPU. The script auto-detects
CUDA and falls back to CPU (slow, but useful for a quick smoke test).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import compute_class_weights
from src.dataset import WasteDataset, CLASSES, SPLITS_PATH
from src.model import build_model

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU detected — this will be slow. "
              "On Colab: Runtime > Change runtime type > T4 GPU.")

    train_ds = WasteDataset("train")
    val_ds = WasteDataset("val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Class-weighted loss — this is where the imbalance handling from
    # step 1 (data.py) actually gets used. Without this, the model would
    # learn to just predict "paper" or "glass" more often since they're
    # more common, and would rarely be penalized enough for missing
    # `trash` to bother learning it well.
    split_df = pd.read_csv(SPLITS_PATH)
    class_weights_dict = compute_class_weights(split_df)
    weights_tensor = torch.tensor([class_weights_dict[c] for c in CLASSES], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    model = build_model(num_classes=len(CLASSES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - start

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} | {elapsed:.1f}s")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })

        # Save the best model by val accuracy, not just the last epoch —
        # the last epoch isn't necessarily the best one, especially once
        # the model starts overfitting.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pt")
            print(f"  -> new best (val_acc={val_acc:.3f}), checkpoint saved")

    with open(CHECKPOINT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
