"""Model zoo: a compact CNN baseline plus pretrained MobileNetV3 and ResNet18."""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


class TrafficSignCNN(nn.Module):
    """A from-scratch CNN baseline suitable for small traffic-sign crops."""

    def __init__(self, num_classes: int, dropout: float = 0.30):
        super().__init__()

        def block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 192),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(model_name: str, num_classes: int, pretrained: bool = True,
                freeze_backbone: bool = False):
    model_name = model_name.lower().strip()
    if model_name == "cnn":
        if freeze_backbone:
            raise ValueError("--freeze-backbone is only meaningful for transfer-learning models.")
        return TrafficSignCNN(num_classes)

    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        try:
            model = models.mobilenet_v3_small(weights=weights)
        except Exception as exc:
            if not pretrained:
                raise
            print(f"WARNING: Could not load ImageNet weights ({exc}). Using random initialization.")
            model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        head = model.classifier
    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            model = models.resnet18(weights=weights)
        except Exception as exc:
            if not pretrained:
                raise
            print(f"WARNING: Could not load ImageNet weights ({exc}). Using random initialization.")
            model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head = model.fc
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose cnn, mobilenet_v3_small, or resnet18.")

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in head.parameters():
            parameter.requires_grad = True
    return model
