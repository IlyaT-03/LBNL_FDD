import torch


def normalize_adj(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    if add_self_loops:
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    deg = adj.sum(dim=1).clamp(min=1e-6)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def corr_graph(data: torch.Tensor, threshold: float = 0.3, top_k: int | None = None) -> torch.Tensor:
    # data: [time, n_nodes]
    x = data - data.mean(dim=0, keepdim=True)
    x = x / x.std(dim=0, keepdim=True).clamp(min=1e-6)

    corr = (x.T @ x) / max(x.size(0) - 1, 1)
    adj = corr.abs()
    adj.fill_diagonal_(0)

    if top_k is not None:
        values, idx = torch.topk(adj, k=top_k, dim=1)
        mask = torch.zeros_like(adj)
        mask.scatter_(1, idx, 1.0)
        adj = adj * mask
    else:
        adj = (adj >= threshold).float() * adj

    return normalize_adj(adj)


def full_graph(n_nodes: int, device=None) -> torch.Tensor:
    adj = torch.ones(n_nodes, n_nodes, device=device)
    return normalize_adj(adj, add_self_loops=False)


def knn_graph(data: torch.Tensor, k: int = 5) -> torch.Tensor:
    # nodes are sensor time-series vectors
    x = data.T  # [n_nodes, time]
    dist = torch.cdist(x, x)
    sim = 1.0 / (1.0 + dist)
    sim.fill_diagonal_(0)

    values, idx = torch.topk(sim, k=k, dim=1)
    adj = torch.zeros_like(sim)
    adj.scatter_(1, idx, values)

    # optional symmetrize
    adj = torch.maximum(adj, adj.T)
    return normalize_adj(adj)