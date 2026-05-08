import torch
import torch.nn as nn

from lbnl_fdd.models.gnn_tam.gsl import GSL


class GCLayer(nn.Module):
    """
    Graph convolution layer.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)

        h = self.dense(X)

        degree = adj.sum(dim=1).clamp(min=1e-12)
        norm = degree.pow(-0.5)

        h = norm[None, :] * adj * norm[:, None] @ h
        return h


class GNN_TAM(nn.Module):
    """
    Model architecture from the paper:
    "Graph Neural Networks with Trainable Adjacency Matrices
    for Fault Diagnosis on Multivariate Sensor Data".
    """
    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        n_classes: int,
        n_gnn: int = 1,
        gsl_type: str = "relu",
        n_hidden: int = 1024,
        alpha: float = 0.1,
        k: int | None = None,
        device: str = "cpu",
    ):
        super(GNN_TAM, self).__init__()

        self.window_size = window_size
        self.n_hidden = n_hidden
        self.n_gnn = n_gnn
        self.device_name = device

        self.register_buffer("idx", torch.arange(n_nodes))
        self.register_buffer(
            "z",
            torch.ones(n_nodes, n_nodes) - torch.eye(n_nodes),
        )

        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for _ in range(n_gnn):
            self.gsl.append(
                GSL(
                    gsl_type=gsl_type,
                    n_nodes=n_nodes,
                    window_size=window_size,
                    alpha=alpha,
                    k=k,
                    device=device,
                )
            )
            self.conv1.append(GCLayer(window_size, n_hidden))
            self.bnorm1.append(nn.BatchNorm1d(n_nodes))
            self.conv2.append(GCLayer(n_hidden, n_hidden))
            self.bnorm2.append(nn.BatchNorm1d(n_nodes))

        self.fc = nn.Linear(n_gnn * n_hidden, n_classes)

        self._last_adj: list[torch.Tensor] = []

        self.to(device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.to(self.idx.device)

        outputs = []
        last_adj = []

        for i in range(self.n_gnn):
            adj = self.gsl[i](self.idx)
            adj = adj * self.z

            h = self.conv1[i](adj, X).relu()
            h = self.bnorm1[i](h)

            skip, _ = torch.min(h, dim=1)

            h = self.conv2[i](adj, h).relu()
            h = self.bnorm2[i](h)

            h, _ = torch.min(h, dim=1)
            h = h + skip

            outputs.append(h)
            last_adj.append(adj.detach())

        self._last_adj = last_adj

        h = torch.cat(outputs, dim=1)
        output = self.fc(h)

        return output

    def get_adj(self):
        return self._last_adj