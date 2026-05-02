import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, N, F]
        adj: [N, N]
        """
        h = self.linear(x)                      # [B, N, out_dim]
        h = torch.einsum("ij,bjf->bif", adj, h)  # graph aggregation
        return h


class SimpleGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        n_classes: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        pool: str = "mean",
        graph_type: str = "corr",   # corr | knn | attention | full
        dropout: float = 0.1,
    ):
        super().__init__()

        if graph_type not in {'full', 'corr', 'attention', 'knn'}:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        if pool not in {"mean", "max", "min"}:
            raise ValueError(f"Unknown pool: {pool}")

        self.graph_type = graph_type
        self.pool = pool
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(window_size, hidden_dim))

        for _ in range(n_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

        # learnable adjacency for attention graph
        if self.graph_type == "attention":
            self.node_emb = nn.Parameter(torch.randn(n_nodes, hidden_dim))
            nn.init.xavier_uniform_(self.node_emb)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def build_attention_adj(self) -> torch.Tensor:
        """
        Build learnable attention-based adjacency matrix.

        Returns:
            adj: [N, N]
        """
        scores = self.node_emb @ self.node_emb.T          # [N, N]
        scores.fill_diagonal_(float("-inf"))              # remove self-attention
        adj = torch.softmax(scores, dim=-1)               # row-normalized attention
        return adj

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:   [B, N, T]
        adj: [N, N] for corr/knn, ignored for attention
        """
        if self.graph_type == "attention":
            adj = self.build_attention_adj()
        elif adj is None:
            raise ValueError("adj must be provided for corr/knn graph")

        adj = adj.to(x.device)
        h = x

        for layer in self.layers:
            h = layer(h, adj)
            h = F.relu(h)
            h = self.dropout(h)

        if self.pool == "mean":
            h = h.mean(dim=1)
        elif self.pool == "max":
            h = h.max(dim=1).values
        elif self.pool == "min":
            h = h.min(dim=1).values

        out = self.classifier(h)
        return out