from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch


@torch.no_grad()
def eval_windows_accuracy_f1(
    model,
    window_ds,
    batch_size=512,
    device="cpu",
    average="macro",
):
    model = model.to(device)
    model.eval()

    loader = DataLoader(
        window_ds,
        batch_size=batch_size,
        shuffle=False,   # КРИТИЧНО
        num_workers=0,
    )

    y_true_all = []
    y_pred_all = []

    for x, y in tqdm(loader):
        x = x.to(device).transpose(1, 2)  # (B, F, T)
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