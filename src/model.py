"""Compact 1-D CNN for raw IMU windows, sized for edge inference."""
import torch.nn as nn


class HARCNN(nn.Module):
    """Input (batch, 9, 128) raw IMU window -> logits (batch, 6)."""

    def __init__(self, in_channels=9, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
