"""
PyTorch Dataset for the waste classifier.

Deliberately thin: all the "which image goes where" logic already lives
in src/data.py (step 1) and is unit-tested there without needing torch
installed at all. This file's only job is turning that manifest into
actual tensors, plus the augmentation strategy.
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SPLITS_PATH = Path(__file__).parent.parent / "data" / "splits.csv"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str) -> transforms.Compose:
    """
    Two different pipelines for train vs. val/test.

    Train gets heavier augmentation than usual on purpose: this dataset
    is 2,500 clean studio photos on plain backgrounds, and the risk
    isn't just standard overfitting — it's the model learning "plain
    white background" as a feature of the *object* rather than an
    artifact of *how this dataset was collected*. Aggressive color/geometry
    jitter is a partial defense against that (a real-world eval set,
    added later, is the actual test of whether it worked).

    Val/test get only the deterministic resize + normalize — we want
    evaluation numbers that reflect the model's real performance, not
    numbers inflated or deflated by random augmentation.
    """
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class WasteDataset(Dataset):
    def __init__(self, split: str, splits_path: Path = SPLITS_PATH):
        assert split in {"train", "val", "test"}
        df = pd.read_csv(splits_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_IDX[row["class"]]
        return image, label
