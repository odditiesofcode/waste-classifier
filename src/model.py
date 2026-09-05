"""
Model definition.

Why transfer learning, not training from scratch: we have 1,768 training
images. A CNN trained from scratch on that little data would badly
overfit. Starting from ImageNet-pretrained weights means the model
already knows general visual features (edges, textures, shapes) and
only has to learn what's specific to our 6 classes — this is the
standard, correct choice for a dataset this size.

ResNet18 specifically: small enough to fine-tune quickly on a free-tier
GPU (or even CPU, slowly, for debugging), while still being a
well-understood, well-documented baseline. Swapping to a bigger backbone
(EfficientNet, ConvNeXt) later is a one-line change if accuracy demands it.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 6, freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        # Only train the final classification head. Faster, and a
        # reasonable choice if you're compute-constrained — but tends to
        # underperform full fine-tuning on datasets this small, since the
        # backbone never adapts to our specific image style at all.
        for param in model.parameters():
            param.requires_grad = False

    # Replace the ImageNet 1000-class head with our 6-class head.
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
