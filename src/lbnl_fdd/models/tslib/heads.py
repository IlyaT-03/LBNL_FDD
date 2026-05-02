import torch
from torch import nn
import torch.nn.functional as F


class FlattenClassificationHead(nn.Module):
    """
    TSLib-style classification head:

    encoder output: (B, T, D)
    mask:           (B, T)
    logits:         (B, C)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(seq_len * d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: (B, T, D)
        x = F.gelu(x)
        x = self.dropout(x)

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)

        x = x.reshape(x.shape[0], -1)
        return self.projection(x)