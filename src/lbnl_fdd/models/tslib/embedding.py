import math

import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in: int, d_model: int):
        super().__init__()

        padding = 1
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        nn.init.kaiming_normal_(
            self.token_conv.weight,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.token_conv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """
    Minimal TSLib-style embedding for classification.

    Uses value embedding + sinusoidal positional embedding.
    Temporal/calendar marks are intentionally omitted.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(
            d_model=d_model,
            max_len=max_len,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        x_mark: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)