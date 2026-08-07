"""The two architectures under comparison."""
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ModelA(nn.Module):
    """Custom scratch CNN. 93,601 params."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.head(x).squeeze(1)


class ModelB(nn.Module):
    """EfficientNet-B0. 4,008,829 params. pretrained=False is the control arm."""

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.net = efficientnet_b0(weights=weights)
        self.net.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)

    def backbone_parameters(self):
        return self.net.features.parameters()

    def head_parameters(self):
        return self.net.classifier.parameters()
