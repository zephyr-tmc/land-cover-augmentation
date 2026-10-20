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

def build_model(num_classes: int, backbone: str = "smallcnn", pretrained: bool = True):
    if backbone == "smallcnn":
        return SmallCNN(num_classes)
    elif backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError("Unsupported backbone")
