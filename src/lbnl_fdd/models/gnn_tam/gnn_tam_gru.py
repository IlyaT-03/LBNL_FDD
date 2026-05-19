import torch
import torch.nn as nn
from lbnl_fdd.models.gnn_tam.gnn import GNN_TAM
from lbnl_fdd.models.gru.gru import GRUClassifier


class GNN_TAM_GRU(nn.Module):
    """
    GNN_TAM + GRU параллельно, по аналогии с LSTM-FCN.
    
    GNN_TAM-ветвь: моделирует связи между сенсорами
    GRU-ветвь:     моделирует динамику во времени
    """
    def __init__(
        self,
        n_nodes: int,
        window_size: int,
        n_classes: int,
        # GNN_TAM параметры
        n_gnn: int = 1,
        gsl_type: str = "tanh",
        n_hidden: int = 1024,
        alpha: float = 0.1,
        k: int | None = None,
        # GRU параметры
        gru_hidden: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        gru_concat_layers: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.window_size = window_size

        # GNN_TAM-ветвь — выход: (B, n_gnn * n_hidden)
        self.gnn = GNN_TAM(
            n_nodes=n_nodes,
            window_size=window_size,
            n_classes=n_classes,   # не используется — перекрываем fc
            n_gnn=n_gnn,
            gsl_type=gsl_type,
            n_hidden=n_hidden,
            alpha=alpha,
            k=k,
            device=device,
        )
        gnn_out_dim = n_gnn * n_hidden

        # GRU-ветвь — выход зависит от concat_layers
        directions = 1
        gru_out_dim = (
            gru_hidden * gru_layers * directions
            if gru_concat_layers
            else gru_hidden * directions
        )
        self.gru_concat_layers = gru_concat_layers
        lstm_dropout = gru_dropout if gru_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=n_nodes,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.gru_drop = nn.Dropout(gru_dropout)

        # перекрываем fc из GNN_TAM
        self.gnn.fc = nn.Identity()

        # общий классификатор
        self.classifier = nn.Linear(gnn_out_dim + gru_out_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, T] — как ожидает GNN_TAM

        # GNN_TAM-ветвь
        gnn_out = self.gnn(x)                          # [B, n_gnn * n_hidden]

        # GRU-ветвь: [B, N, T] → [B, T, N]
        x_seq = x.permute(0, 2, 1)                     # [B, T, N]
        _, h = self.gru(x_seq)                         # h: [layers, B, hidden]
        if self.gru_concat_layers:
            h = h.permute(1, 0, 2).reshape(h.size(1), -1)  # [B, layers*hidden]
        else:
            h = h[-1]                                       # [B, hidden]
        gru_out = self.gru_drop(h)                     # [B, gru_out_dim]

        # конкатенация и классификация
        out = torch.cat([gnn_out, gru_out], dim=1)
        return self.classifier(out)

    def get_adj(self):
        return self.gnn.get_adj()