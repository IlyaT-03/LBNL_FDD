import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP baseline for windowed multivariate time series.

    Expected input:
        X of shape (batch_size, n_nodes, window_size)
    or
        X of shape (n_nodes, window_size) for a single sample.

    The model flattens each window into a vector of size
        n_nodes * window_size
    and applies a feed-forward network for classification.
    """
    def __init__(
            self,
            n_nodes: int,
            window_size: int,
            n_classes: int,
            n_hidden: int = 1024,
            n_layers: int = 2,
            dropout: float = 0.3,
            device: str = 'cpu'
            ):
        super().__init__()

        self.n_nodes = n_nodes
        self.window_size = window_size
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = device

        input_dim = n_nodes * window_size

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, n_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = n_hidden

        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        X = X.to(self.device)

        if X.dim() == 2:
            X = X.unsqueeze(0)

        X = X.reshape(X.size(0), -1)
        output = self.net(X)

        return output