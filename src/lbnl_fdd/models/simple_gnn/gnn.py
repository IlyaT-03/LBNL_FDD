import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = torch.einsum("ij,bjf->bif", adj, h)
        return h


class SimpleGNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        n_classes: int,
        hidden_dim: int = 1024,
        graph_type: str = "corr",
        dropout: float = 0.0,
        alpha: float = 0.1,
    ):
        super().__init__()

        if graph_type not in {"corr", "knn", "attention", "full"}:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        self.graph_type = graph_type
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)

        self.gcn1 = GCNLayer(window_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(n_nodes)

        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(n_nodes)

        if graph_type == "attention":
            self.attn = nn.Linear(window_size, hidden_dim, bias=False)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def build_attention_adj(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, T]
        dynamic adjacency from current batch
        """
        z = self.attn(x)  # [B, N, H]

        scores = torch.matmul(z, z.transpose(1, 2))  # [B, N, N]
        scores = scores / (z.size(-1) ** 0.5)
        scores = torch.tanh(self.alpha * scores)

        n_nodes = scores.size(-1)
        eye = torch.eye(n_nodes, device=scores.device, dtype=torch.bool)

        scores = scores.masked_fill(
            eye.unsqueeze(0),
            float("-inf"),
        )

        adj = torch.softmax(scores, dim=-1)
        return adj

    def graph_conv(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.dim() == 2:
            return torch.einsum("ij,bjf->bif", adj, h)

        if adj.dim() == 3:
            return torch.einsum("bij,bjf->bif", adj, h)

        raise ValueError(f"Expected adj dim 2 or 3, got {adj.dim()}")

    def min_readout(self, h: torch.Tensor) -> torch.Tensor:
        return h.min(dim=1).values

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, N, T]
        adj:
          [N, N] for corr/knn/full
          None for attention
        """
        if self.graph_type == "attention":
            adj = self.build_attention_adj(x)
        elif adj is None:
            raise ValueError("adj must be provided for corr/knn/full graph")

        adj = adj.to(x.device)

        h1 = self.gcn1.linear(x)
        h1 = self.graph_conv(h1, adj)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        r1 = self.min_readout(h1)

        h2 = self.gcn2.linear(h1)
        h2 = self.graph_conv(h2, adj)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        r2 = self.min_readout(h2)

        h = r1 + r2
        return self.classifier(h)