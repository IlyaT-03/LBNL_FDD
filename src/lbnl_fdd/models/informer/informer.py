import torch
from torch import nn

from lbnl_fdd.models.tslib import (
    AttentionLayer,
    ConvLayer,
    DataEmbedding,
    Encoder,
    EncoderLayer,
    FlattenClassificationHead,
    ProbAttention,
    ensure_btf,
    make_padding_mask,
)


class InformerClassifier(nn.Module):
    """
    Informer encoder-only classifier for window classification.

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
        distil: bool = False,
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

        self.enc_embedding = DataEmbedding(
            c_in=n_features,
            d_model=d_model,
            dropout=dropout,
        )

        attn_layers = [
            EncoderLayer(
                attention=AttentionLayer(
                    attention=ProbAttention(
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
        ]

        conv_layers = (
            [ConvLayer(d_model) for _ in range(e_layers - 1)]
            if distil and e_layers > 1
            else None
        )

        self.encoder = Encoder(
            attn_layers=attn_layers,
            conv_layers=conv_layers,
            norm_layer=nn.LayerNorm(d_model),
        )

        final_seq_len = seq_len
        if distil and e_layers > 1:
            for _ in range(e_layers - 1):
                final_seq_len = (final_seq_len + 1) // 2

        self.head = FlattenClassificationHead(
            seq_len=final_seq_len,
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

        x = self.enc_embedding(x, x_mark=None)
        x, attns = self.encoder(
            x,
            attn_mask=None,
            tau=None,
            delta=None,
        )

        logits = self.head(x, padding_mask=padding_mask)

        if self.output_attention:
            return logits, attns

        return logits