import torch
from torch import nn

from lbnl_fdd.models.tslib import (
    AttentionLayer,
    DSAttention,
    DataEmbedding,
    Encoder,
    EncoderLayer,
    FlattenClassificationHead,
    Projector,
    ensure_btf,
    make_padding_mask,
)


class NonstationaryTransformerClassifier(nn.Module):
    """
    Non-stationary Transformer encoder-only classifier for window classification.

    Input:
        x: (B, T, F) or (B, F, T)

    Output:
        logits: (B, C)
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        factor: int = 5,
        activation: str = "gelu",
        p_hidden_dims: list[int] | None = None,
        p_hidden_layers: int = 2,
        output_attention: bool = False,
    ):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

        self.n_features = n_features
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.output_attention = output_attention

        if p_hidden_dims is None:
            p_hidden_dims = [128, 128]

        self.enc_embedding = DataEmbedding(
            c_in=n_features,
            d_model=d_model,
            dropout=dropout,
        )

        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    attention=AttentionLayer(
                        attention=DSAttention(
                            mask_flag=False,
                            factor=factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            conv_layers=None,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.tau_learner = Projector(
            enc_in=n_features,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=1,
        )

        self.delta_learner = Projector(
            enc_in=n_features,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=seq_len,
        )

        self.head = FlattenClassificationHead(
            seq_len=seq_len,
            d_model=d_model,
            n_classes=n_classes,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor | None]]:
        x = ensure_btf(x, n_features=self.n_features)   # -> (B, T, F)
        padding_mask = make_padding_mask(x)             # -> (B, T)

        # De-stationary normalization
        means = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - means

        stdev = torch.sqrt(
            torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()

        x_norm = x_centered / stdev

        tau = self.tau_learner(x, stdev).exp().clamp(max=80.0).squeeze(-1)   # (B,)
        delta = self.delta_learner(x, means)                                  # (B, T)

        x_embed = self.enc_embedding(x_norm, x_mark=None)

        x_enc, attns = self.encoder(
            x_embed,
            attn_mask=None,
            tau=tau,
            delta=delta,
        )

        logits = self.head(x_enc, padding_mask=padding_mask)

        if self.output_attention:
            return logits, attns

        return logits