import torch
from torch import nn
import torch.nn.functional as F

from lbnl_fdd.models.tslib import ensure_btf


class CNN1DClassifier(nn.Module):
    """
    Depthwise 1D CNN classifier.

    Input:
        x: (B, T, F) or (B, F, T)

    Output:
        logits: (B, C)
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        n_classes: int,
        conv1_multiplier: int = 4,
        conv2_multiplier: int = 16,
        kernel_size: int = 5,
        stride: int = 5,
        pool_size: int = 2,
        pool_stride: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.n_classes = n_classes

        conv1_channels = n_features * conv1_multiplier
        conv2_channels = n_features * conv2_multiplier

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=conv1_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=n_features,
        )
        self.pool1 = nn.MaxPool1d(
            kernel_size=pool_size,
            stride=pool_stride,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=conv1_channels,
        )
        self.pool2 = nn.MaxPool1d(
            kernel_size=pool_size,
            stride=pool_stride,
        )

        flattened_dim = self._infer_flattened_dim()

        self.fc1 = nn.Linear(flattened_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def _infer_flattened_dim(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_features, self.window_size)
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            return int(x.reshape(1, -1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btf(x, n_features=self.n_features)  # (B, T, F)
        x = x.transpose(1, 2)                          # (B, F, T)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)