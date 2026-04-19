from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlockV1(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_kernels: int = 6):
        super().__init__()
        if num_kernels < 1:
            raise ValueError("num_kernels must be >= 1")

        kernel_sizes = [2 * i + 1 for i in range(num_kernels)]
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, k),
                    padding=(0, k // 2),
                )
                for k in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [conv(x) for conv in self.branches]
        return torch.stack(outs, dim=-1).mean(dim=-1)


def _compute_topk_periods(x: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, T, C)
    returns:
        periods: (K,)
        weights: (B, K)
    """
    xf = torch.fft.rfft(x, dim=1)
    freq_amplitude = xf.abs().mean(dim=0).mean(dim=-1)

    if freq_amplitude.numel() > 0:
        freq_amplitude[0] = 0

    k = min(top_k, max(freq_amplitude.numel() - 1, 1))
    _, indices = torch.topk(freq_amplitude, k=k)

    seq_len = x.size(1)
    periods: List[int] = []
    for idx in indices:
        freq_idx = int(idx.item())
        if freq_idx <= 0:
            period = seq_len
        else:
            period = max(seq_len // freq_idx, 1)
        periods.append(period)

    periods_tensor = torch.tensor(periods, device=x.device, dtype=torch.long)
    batch_weights = xf.abs().mean(dim=-1)[:, indices]
    return periods_tensor, batch_weights


class TimesBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, top_k: int, num_kernels: int = 6):
        super().__init__()
        self.top_k = top_k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )
        self.norm = nn.LayerNorm(d_model)

    def _reshape_by_period(self, x: torch.Tensor, period: int):
        bsz, seq_len, d_model = x.shape

        if seq_len % period != 0:
            padded_len = ((seq_len // period) + 1) * period
            pad_size = padded_len - seq_len
            pad_tensor = torch.zeros(
                bsz, pad_size, d_model, device=x.device, dtype=x.dtype
            )
            x = torch.cat([x, pad_tensor], dim=1)
        else:
            padded_len = seq_len

        x = x.reshape(bsz, padded_len // period, period, d_model)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x, padded_len

    def _restore_shape(self, x: torch.Tensor, padded_len: int, original_len: int):
        bsz, d_model, blocks, period = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(bsz, padded_len, d_model)[:, :original_len, :]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        periods, period_weights = _compute_topk_periods(x, self.top_k)

        outputs = []
        for period in periods.tolist():
            reshaped, padded_len = self._reshape_by_period(x, period)
            out = self.conv(reshaped)
            out = self._restore_shape(out, padded_len, seq_len)
            outputs.append(out)

        stacked = torch.stack(outputs, dim=-1)  # (B, T, D, K)
        weights = F.softmax(period_weights, dim=1).view(bsz, 1, 1, -1)
        mixed = (stacked * weights).sum(dim=-1)

        return self.norm(mixed + x)


class TimesNetClassifier(nn.Module):
    """
    Input:  (B, T, F)
    Output: (B, n_classes)
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        n_classes: int,
        d_model: int = 64,
        d_ff: int = 128,
        e_layers: int = 2,
        top_k: int = 3,
        num_kernels: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.n_classes = int(n_classes)

        self.value_embedding = nn.Linear(self.n_features, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.window_size, d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    top_k=top_k,
                    num_kernels=num_kernels,
                )
                for _ in range(e_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape (B, T, F), got {tuple(x.shape)}")

        _, seq_len, n_features = x.shape
        if n_features != self.n_features:
            raise ValueError(
                f"Expected n_features={self.n_features}, got {n_features}"
            )
        if seq_len != self.window_size:
            raise ValueError(
                f"Expected window_size={self.window_size}, got {seq_len}"
            )

        x = self.value_embedding(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits