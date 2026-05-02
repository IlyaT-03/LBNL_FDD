import torch
from torch import nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Informer distillation layer.
    Optional for classification; useful only if you want distil=True.
    """

    def __init__(self, c_in: int):
        super().__init__()

        self.down_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.down_conv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        d_ff = d_ff or 4 * d_model

        self.attention = attention
        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ):
        new_x, attn = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )

        x = x + self.dropout(new_x)

        y = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(
        self,
        attn_layers: list[nn.Module],
        conv_layers: list[nn.Module] | None = None,
        norm_layer: nn.Module | None = None,
    ):
        super().__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers)
            if conv_layers is not None
            else None
        )
        self.norm = norm_layer

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ):
        attns = []

        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(
                self.attn_layers[:-1],
                self.conv_layers,
            ):
                x, attn = attn_layer(
                    x,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=delta,
                )
                x = conv_layer(x)
                attns.append(attn)

            x, attn = self.attn_layers[-1](
                x,
                attn_mask=attn_mask,
                tau=tau,
                delta=delta,
            )
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(
                    x,
                    attn_mask=attn_mask,
                    tau=tau,
                    delta=delta,
                )
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns