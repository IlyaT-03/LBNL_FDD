import math

import torch
from torch import nn


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag: bool = False,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ):
        del tau, delta

        batch_size, query_len, n_heads, head_dim = queries.shape
        _, key_len, _, _ = keys.shape

        scale = self.scale or 1.0 / math.sqrt(head_dim)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), -torch.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return output.contiguous(), attn

        return output.contiguous(), None


class DSAttention(nn.Module):
    """
    De-stationary attention from Non-stationary Transformer.
    """

    def __init__(
        self,
        mask_flag: bool = False,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        del factor

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ):
        batch_size, query_len, n_heads, head_dim = queries.shape
        _, key_len, _, _ = keys.shape

        scale = self.scale or 1.0 / math.sqrt(head_dim)

        tau = 1.0 if tau is None else tau.view(tau.shape[0], 1, 1, 1)
        delta = 0.0 if delta is None else delta.view(delta.shape[0], 1, 1, delta.shape[-1])

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        scores = scores * tau + delta

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), -torch.inf)

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn, values)

        if self.output_attention:
            return output.contiguous(), attn

        return output.contiguous(), None


class ProbAttention(nn.Module):
    """
    ProbSparse attention from Informer.

    This is a compact version suitable for encoder/classification usage.
    """

    def __init__(
        self,
        mask_flag: bool = False,
        factor: int = 5,
        scale: float | None = None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()

        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_qk(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        sample_k: int,
        n_top: int,
    ):
        # q, k: (B, H, L, D)
        batch_size, n_heads, key_len, head_dim = k.shape
        _, _, query_len, _ = q.shape

        k_expand = k.unsqueeze(-3).expand(
            batch_size,
            n_heads,
            query_len,
            key_len,
            head_dim,
        )

        index_sample = torch.randint(
            key_len,
            (query_len, sample_k),
            device=q.device,
        )

        k_sample = k_expand[
            :,
            :,
            torch.arange(query_len, device=q.device).unsqueeze(1),
            index_sample,
            :,
        ]

        q_k_sample = torch.matmul(
            q.unsqueeze(-2),
            k_sample.transpose(-2, -1),
        ).squeeze(-2)

        sparsity_measure = q_k_sample.max(-1)[0] - torch.div(
            q_k_sample.sum(-1),
            key_len,
        )

        m_top = sparsity_measure.topk(n_top, sorted=False)[1]

        q_reduce = q[
            torch.arange(batch_size, device=q.device)[:, None, None],
            torch.arange(n_heads, device=q.device)[None, :, None],
            m_top,
            :,
        ]

        q_k = torch.matmul(q_reduce, k.transpose(-2, -1))

        return q_k, m_top

    def _get_initial_context(
        self,
        v: torch.Tensor,
        query_len: int,
    ) -> torch.Tensor:
        # v: (B, H, L, D)
        batch_size, n_heads, value_len, head_dim = v.shape

        if not self.mask_flag:
            context = v.mean(dim=-2)
            context = context.unsqueeze(-2).expand(
                batch_size,
                n_heads,
                query_len,
                head_dim,
            ).clone()
            return context

        if query_len != value_len:
            raise ValueError("Causal ProbAttention requires query_len == value_len")

        return v.cumsum(dim=-2)

    def _update_context(
        self,
        context: torch.Tensor,
        v: torch.Tensor,
        scores: torch.Tensor,
        index: torch.Tensor,
        query_len: int,
        attn_mask: torch.Tensor | None,
    ):
        batch_size, n_heads, value_len, head_dim = v.shape

        if self.mask_flag and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), -torch.inf)

        attn = torch.softmax(scores, dim=-1)

        context[
            torch.arange(batch_size, device=v.device)[:, None, None],
            torch.arange(n_heads, device=v.device)[None, :, None],
            index,
            :,
        ] = torch.matmul(attn, v).type_as(context)

        if self.output_attention:
            full_attn = torch.ones(
                (batch_size, n_heads, query_len, value_len),
                device=v.device,
                dtype=attn.dtype,
            ) / value_len

            full_attn[
                torch.arange(batch_size, device=v.device)[:, None, None],
                torch.arange(n_heads, device=v.device)[None, :, None],
                index,
                :,
            ] = attn

            return context, full_attn

        return context, None

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ):
        del tau, delta

        # Input: (B, L, H, D)
        batch_size, query_len, n_heads, head_dim = queries.shape
        _, key_len, _, _ = keys.shape

        q = queries.transpose(2, 1)  # (B, H, L_Q, D)
        k = keys.transpose(2, 1)     # (B, H, L_K, D)
        v = values.transpose(2, 1)   # (B, H, L_V, D)

        sample_k = min(self.factor * math.ceil(math.log(key_len)), key_len)
        n_top = min(self.factor * math.ceil(math.log(query_len)), query_len)

        scores_top, index = self._prob_qk(
            q=q,
            k=k,
            sample_k=sample_k,
            n_top=n_top,
        )

        scale = self.scale or 1.0 / math.sqrt(head_dim)
        scores_top = scores_top * scale

        context = self._get_initial_context(v, query_len)
        context, attn = self._update_context(
            context=context,
            v=v,
            scores=scores_top,
            index=index,
            query_len=query_len,
            attn_mask=attn_mask,
        )

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        n_heads: int,
        d_keys: int | None = None,
        d_values: int | None = None,
    ):
        super().__init__()

        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        tau: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ):
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        n_heads = self.n_heads

        queries = self.query_projection(queries).view(
            batch_size,
            query_len,
            n_heads,
            -1,
        )
        keys = self.key_projection(keys).view(
            batch_size,
            key_len,
            n_heads,
            -1,
        )
        values = self.value_projection(values).view(
            batch_size,
            key_len,
            n_heads,
            -1,
        )

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta,
        )

        out = out.view(batch_size, query_len, -1)
        return self.out_projection(out), attn