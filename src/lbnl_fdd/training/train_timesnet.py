from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from lbnl_fdd.training.evaluate_timesnet import eval_windows_accuracy_f1_timesnet


def train_timesnet(
    model,
    train_ds,
    val_ds=None,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    save_dir: str | None = None,
    save_best: bool = True,
    average: str = "macro",
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device != "cpu"),
    )

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        save_path = None

    history = {
        "train_loss": [],
        "val_accuracy": [],
        f"val_f1_{average}": [],
    }

    best_score = float("-inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for x, y in progress:
            x = x.to(device).float()   # (B, T, F)
            y = y.to(device).long()

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_size_curr = x.size(0)
            running_loss += loss.item() * batch_size_curr
            n_samples += batch_size_curr

            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running_loss / max(n_samples, 1)
        history["train_loss"].append(epoch_loss)

        msg = f"Epoch {epoch}/{epochs} | train_loss={epoch_loss:.4f}"

        if val_ds is not None:
            val_metrics, _, _ = eval_windows_accuracy_f1_timesnet(
                model=model,
                window_ds=val_ds,
                batch_size=batch_size,
                device=device,
                average=average,
            )

            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics[f"f1_{average}"]

            history["val_accuracy"].append(val_acc)
            history[f"val_f1_{average}"].append(val_f1)

            msg += f" | val_acc={val_acc:.4f} | val_f1_{average}={val_f1:.4f}"

            score = val_f1
            if save_best and save_path is not None and score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_score": best_score,
                        "history": history,
                    },
                    save_path / "best_model.pt",
                )

        if save_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                save_path / "last_model.pt",
            )

            with open(save_path / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        print(msg)

    if val_ds is not None and save_best and best_epoch != -1:
        print(f"Best epoch: {best_epoch}, best val_f1_{average}: {best_score:.4f}")

    return history