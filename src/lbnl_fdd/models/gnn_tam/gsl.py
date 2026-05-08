import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_topk(adj, k):
    if k is None:
        return adj

    mask = torch.zeros_like(adj)
    _, idx = adj.topk(k, dim=1)
    mask.scatter_(1, idx, 1.0)
    return adj * mask


def _apply_topk_abs(adj, k):
    if k is None:
        return adj

    mask = torch.zeros_like(adj)
    _, idx = adj.abs().topk(k, dim=1)
    mask.scatter_(1, idx, 1.0)
    return adj * mask


# A = ReLU(W)
class Graph_ReLu_W(nn.Module):
    def __init__(self, n_nodes, k, device):
        super(Graph_ReLu_W, self).__init__()
        self.k = k
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device))

    def forward(self, idx):
        adj = F.relu(self.A)
        return _apply_topk(adj, self.k)


# A = Tanh(alpha * W)
class Graph_Tanh_W(nn.Module):
    def __init__(self, n_nodes, alpha, k, device):
        super(Graph_Tanh_W, self).__init__()
        self.alpha = alpha
        self.k = k
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device))

    def forward(self, idx):
        adj = torch.tanh(self.alpha * self.A)

        # важно: top-k по модулю, чтобы сохранить сильные отрицательные связи
        return _apply_topk_abs(adj, self.k)


# A for Directed graphs
class Graph_Directed_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k, device):
        super(Graph_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k

        self.e1 = nn.Embedding(n_nodes, window_size).to(device)
        self.e2 = nn.Embedding(n_nodes, window_size).to(device)
        self.l1 = nn.Linear(window_size, window_size).to(device)
        self.l2 = nn.Linear(window_size, window_size).to(device)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha * self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha * self.l2(self.e2(idx)))

        adj = F.relu(torch.tanh(self.alpha * torch.mm(m1, m2.transpose(1, 0))))
        return _apply_topk(adj, self.k)


# A for Uni-directed graphs
class Graph_Uni_Directed_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k, device):
        super(Graph_Uni_Directed_A, self).__init__()
        self.alpha = alpha
        self.k = k

        self.e1 = nn.Embedding(n_nodes, window_size).to(device)
        self.e2 = nn.Embedding(n_nodes, window_size).to(device)
        self.l1 = nn.Linear(window_size, window_size).to(device)
        self.l2 = nn.Linear(window_size, window_size).to(device)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha * self.l1(self.e1(idx)))
        m2 = torch.tanh(self.alpha * self.l2(self.e2(idx)))

        adj = F.relu(
            torch.tanh(
                self.alpha * (
                    torch.mm(m1, m2.transpose(1, 0))
                    - torch.mm(m2, m1.transpose(1, 0))
                )
            )
        )
        return _apply_topk(adj, self.k)


# A for Undirected graphs
class Graph_Undirected_A(nn.Module):
    def __init__(self, n_nodes, window_size, alpha, k, device):
        super(Graph_Undirected_A, self).__init__()
        self.alpha = alpha
        self.k = k

        self.e1 = nn.Embedding(n_nodes, window_size).to(device)
        self.l1 = nn.Linear(window_size, window_size).to(device)

    def forward(self, idx):
        m1 = torch.tanh(self.alpha * self.l1(self.e1(idx)))

        adj = F.relu(torch.tanh(self.alpha * torch.mm(m1, m1.transpose(1, 0))))
        return _apply_topk(adj, self.k)


class GSL(nn.Module):
    """
    Graph structure learning block.
    """
    def __init__(
        self,
        gsl_type,
        n_nodes,
        window_size,
        alpha,
        k,
        device,
    ):
        super(GSL, self).__init__()

        if gsl_type == "relu":
            self.gsl_layer = Graph_ReLu_W(n_nodes, k, device)
        elif gsl_type == "tanh":
            self.gsl_layer = Graph_Tanh_W(n_nodes, alpha, k, device)
        elif gsl_type == "directed":
            self.gsl_layer = Graph_Directed_A(n_nodes, window_size, alpha, k, device)
        elif gsl_type == "unidirected":
            self.gsl_layer = Graph_Uni_Directed_A(n_nodes, window_size, alpha, k, device)
        elif gsl_type == "undirected":
            self.gsl_layer = Graph_Undirected_A(n_nodes, window_size, alpha, k, device)
        else:
            raise ValueError(
                f"Wrong name of graph structure learning layer: {gsl_type}"
            )

    def forward(self, idx):
        return self.gsl_layer(idx)