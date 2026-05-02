import torch


def ensure_btf(x: torch.Tensor, n_features: int | None = None) -> torch.Tensor:
    """
    Ensures input format is (B, T, F).

    If x is (B, F, T) and n_features is provided, it is transposed.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got shape {tuple(x.shape)}")

    if n_features is not None:
        if x.shape[-1] == n_features:
            return x
        if x.shape[1] == n_features:
            return x.transpose(1, 2)

    return x


def make_padding_mask(
    x: torch.Tensor,
    valid_value: float = 1.0,
) -> torch.Tensor:
    """
    Creates a classification mask of shape (B, T).

    In this project sliding windows have fixed length, so by default all
    timesteps are valid.
    """
    if x.dim() != 3:
        raise ValueError(f"Expected input shape (B, T, F), got {tuple(x.shape)}")

    batch_size, seq_len, _ = x.shape
    return torch.full(
        (batch_size, seq_len),
        fill_value=valid_value,
        device=x.device,
        dtype=x.dtype,
    )