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
        h = self.linear(x)
        h = torch.einsum("ij,bjf->bif", adj, h)
        return h


class SimpleGNN(nn.Module):
    """
    Simplified GNN baseline close to the paper:
    GCN -> BatchNorm -> ReLU -> min readout
    GCN -> BatchNorm -> ReLU -> min readout
    readout1 + readout2 -> FC
    """

    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        n_classes: int,
        hidden_dim: int = 64,
        graph_type: str = "corr",
        dropout: float = 0.0,
        alpha: float = 0.1,
    ):
        super().__init__()

        if graph_type not in {"corr", "knn", "attention", "full"}:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        self.n_nodes = n_nodes
        self.graph_type = graph_type
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha

        self.gcn1 = GCNLayer(window_size, hidden_dim)
        self.bn1 = nn.BatchNorm1d(n_nodes)

        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(n_nodes)

        if graph_type == "attention":
            self.node_emb = nn.Parameter(torch.empty(n_nodes, hidden_dim))
            nn.init.xavier_uniform_(self.node_emb)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def build_attention_adj(self) -> torch.Tensor:
        scores = self.node_emb @ self.node_emb.T
        scores = torch.tanh(self.alpha * scores)
        scores.fill_diagonal_(float("-inf"))
        adj = torch.softmax(scores, dim=-1)
        return adj

    def min_readout(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, N, H]
        returns: [B, H]
        """
        return h.min(dim=1).values

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        """
        x:   [B, N, T]
        adj: [N, N] for corr/knn/full, None for attention
        """
        if self.graph_type == "attention":
            adj = self.build_attention_adj()
        elif adj is None:
            raise ValueError("adj must be provided for corr/knn/full graph")

        adj = adj.to(x.device)

        h1 = self.gcn1(x, adj)
        h1 = self.bn1(h1)
        h1 = F.relu(h1)
        h1 = self.dropout(h1)
        r1 = self.min_readout(h1)

        h2 = self.gcn2(h1, adj)
        h2 = self.bn2(h2)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        r2 = self.min_readout(h2)

        h = r1 + r2
        out = self.classifier(h)

        return out