from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def eval_windows_accuracy_f1_timesnet(
    model,
    window_ds,
    batch_size: int = 512,
    device: str = "cpu",
    average: str = "macro",
):
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        window_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    y_true_all = []
    y_pred_all = []

    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device).float()   # (B, T, F)
        y = y.to(device).long()

        logits = model(x)
        preds = logits.argmax(dim=1)

        y_true_all.append(y.cpu().numpy())
        y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        f"f1_{average}": f1_score(y_true, y_pred, average=average),
        "n_windows": len(y_true),
    }
    return metrics, y_true, y_pred