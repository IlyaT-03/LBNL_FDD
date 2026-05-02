import torch
from torch import nn


class Projector(nn.Module):
    """
    MLP projector used in Non-stationary Transformer to produce tau/delta.

    Input:
        x:     (B, T, F)
        stats: (B, 1, F)

    Output:
        tau:   (B, 1) or
        delta: (B, T)
    """

    def __init__(
        self,
        enc_in: int,
        seq_len: int,
        hidden_dims: list[int],
        hidden_layers: int,
        output_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")

        padding = 1 if torch.__version__ >= "1.5.0" else 2

        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        layers: list[nn.Module] = []
        input_dim = 2 * enc_in

        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            layers.append(nn.ReLU())
            input_dim = hidden_dims[i]

        layers.append(nn.Linear(input_dim, output_dim, bias=False))

        self.backbone = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        stats: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        x = self.series_conv(x)
        x = torch.cat([x, stats], dim=1)
        x = x.view(batch_size, -1)

        return self.backbone(x)