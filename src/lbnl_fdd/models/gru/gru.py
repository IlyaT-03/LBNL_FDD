import torch
from torch import nn

from lbnl_fdd.models.tslib import ensure_btf


class GRUClassifier(nn.Module):
    """
    GRU classifier for sliding-window classification.

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
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.n_features = n_features
        self.window_size = window_size
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        directions = 2 if bidirectional else 1
        self.linear1 = nn.Linear(hidden_dim * num_layers * directions, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_btf(x, n_features=self.n_features)

        _, h = self.gru(x)
        # h: (num_layers * directions, B, hidden_dim)

        h = h.permute(1, 0, 2)
        h = h.reshape(h.size(0), -1)

        x = self.linear1(h)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.linear2(x)