# model.py
import torch
import torch.nn as nn
from torchvision import models

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1, bias=False), nn.BatchNorm2d(128),nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1, bias=False),nn.BatchNorm2d(256),nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.head(x)
        return x


class SmallCNNPP(nn.Module):
    """
    SmallCNN++：在保持“轻量 + 对域差异敏感”的前提下提升容量与稳健性。
    - 更宽/更深一点的 conv stage
    - 继续使用 Global Average Pooling（AdaptiveAvgPool2d(1)）
    - 更强的 Dropout
    """

    def __init__(self, num_classes: int, dropout: float = 0.35, width: int = 48):
        super().__init__()

        c1 = width          # 48
        c2 = width * 2      # 96
        c3 = width * 3      # 144
        c4 = width * 4      # 192

        def block(cin: int, cout: int):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        # 多一层 stage + 更宽通道
        self.features = nn.Sequential(
            block(3, c1),
            block(c1, c1),
            nn.MaxPool2d(2),

            block(c1, c2),
            block(c2, c2),
            nn.MaxPool2d(2),

            block(c2, c3),
            block(c3, c3),
            nn.MaxPool2d(2),

            block(c3, c4),
            block(c4, c4),
        )

        # Global Average Pooling + Dropout
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x

def build_model(num_classes: int, backbone: str = "smallcnn", pretrained: bool = True):
    if backbone == "smallcnn":
        return SmallCNN(num_classes)
    elif backbone == "smallcnnpp":
        return SmallCNNPP(num_classes)
    elif backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError("Unsupported backbone")
