import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from lbnl_fdd.data.sliding_window import SlidingWindowDataset
from lbnl_fdd.data.selected_window import SelectedWindowsDataset
from lbnl_fdd.models.simple_gnn import SimpleGNN
from lbnl_fdd.models.simple_gnn.graph_builders import corr_graph, knn_graph, full_graph
from lbnl_fdd.training.evaluate import eval_windows_accuracy_f1
from lbnl_fdd.training.train_gnn_tam import train_gnn


class SimpleGNNWithAdj(torch.nn.Module):
    def __init__(self, model: SimpleGNN, adj: torch.Tensor | None):
        super().__init__()
        self.model = model

        if adj is not None:
            self.register_buffer("adj", adj)
        else:
            self.adj = None

    def forward(self, x):
        return self.model(x, self.adj)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimpleGNN baseline")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. SDAHU")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--save_root", type=str, default="outputs/runs")
    parser.add_argument("--run_name", type=str, default="simple_gnn_run1")

    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--graph_type",
        type=str,
        default="corr",
        choices=["corr", "knn", "attention", "full"],
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.3)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--average", type=str, default="macro")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--eval_train", action="store_true")
    parser.add_argument("--eval_test", action="store_true")

    parser.add_argument("--train_windows_file", type=str, default=None)
    parser.add_argument("--val_windows_file", type=str, default=None)
    parser.add_argument("--test_windows_file", type=str, default=None)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir: Path, split: str):
    df = pd.read_csv(data_dir / f"{split}_df.csv", index_col=[0, 1])
    target = pd.read_csv(data_dir / f"{split}_target.csv", index_col=[0, 1]).iloc[:, 0]
    return df, target


def build_dataset(
    df: pd.DataFrame,
    target: pd.Series,
    window_size: int,
    stride: int,
    windows_file: str | None,
    split_name: str,
):
    if windows_file is None:
        return SlidingWindowDataset(
            df=df,
            target=target,
            window_size=window_size,
            stride=stride,
        )

    print(f"Using selected {split_name} windows: {windows_file}")
    windows_df = pd.read_json(windows_file, lines=True)

    return SelectedWindowsDataset(
        df=df,
        windows_df=windows_df,
    )


def build_adj(args, train_df: pd.DataFrame, device: str):
    if args.graph_type == "attention":
        return None

    n_nodes = int(train_df.shape[1])
    train_data = torch.tensor(train_df.values, dtype=torch.float32)

    if args.graph_type == "corr":
        print(f"Building correlation graph: threshold={args.threshold}, top_k={args.k}")
        adj = corr_graph(
            data=train_data,
            threshold=args.threshold,
            top_k=args.k,
        )

    elif args.graph_type == "knn":
        print(f"Building KNN graph: k={args.k}")
        adj = knn_graph(
            data=train_data,
            k=args.k,
        )

    elif args.graph_type == "full":
        print("Building full graph")
        adj = full_graph(
            n_nodes=n_nodes,
        )

    else:
        raise ValueError(f"Unknown graph_type: {args.graph_type}")

    return adj.to(device)


def evaluate_and_save(
    model,
    dataset,
    batch_size: int,
    device: str,
    average: str,
    save_dir: Path,
    split_name: str,
):
    metrics, y_true, y_pred = eval_windows_accuracy_f1(
        model=model,
        window_ds=dataset,
        batch_size=batch_size,
        device=device,
        average=average,
    )

    print(f"{split_name.capitalize()} metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(save_dir / f"{split_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    np.save(save_dir / f"{split_name}_y_true.npy", y_true)
    np.save(save_dir / f"{split_name}_y_pred.npy", y_pred)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"

    data_dir = Path(args.data_root) / args.dataset
    save_dir = Path(args.save_root) / args.dataset / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Using device: {device}")
    print(f"Loading dataset from: {data_dir}")

    train_df, train_target = load_split(data_dir, "train")
    val_df, val_target = load_split(data_dir, "val")
    test_df, test_target = load_split(data_dir, "test")

    if args.standardize:
        print("Applying StandardScaler...")
        scaler = StandardScaler()

        train_df = pd.DataFrame(
            scaler.fit_transform(train_df),
            index=train_df.index,
            columns=train_df.columns,
        )

        val_df = pd.DataFrame(
            scaler.transform(val_df),
            index=val_df.index,
            columns=val_df.columns,
        )

        test_df = pd.DataFrame(
            scaler.transform(test_df),
            index=test_df.index,
            columns=test_df.columns,
        )

    train_ds = build_dataset(
        df=train_df,
        target=train_target,
        window_size=args.window_size,
        stride=args.stride,
        windows_file=args.train_windows_file,
        split_name="train",
    )

    val_ds = build_dataset(
        df=val_df,
        target=val_target,
        window_size=args.window_size,
        stride=args.stride,
        windows_file=args.val_windows_file,
        split_name="val",
    )

    test_ds = build_dataset(
        df=test_df,
        target=test_target,
        window_size=args.window_size,
        stride=args.stride,
        windows_file=args.test_windows_file,
        split_name="test",
    )

    n_nodes = int(train_df.shape[1])
    n_classes = int(train_target.nunique())

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Test windows: {len(test_ds)}")
    print(f"Num features: {n_nodes}")
    print(f"Num classes: {n_classes}")
    print(f"Graph type: {args.graph_type}")

    adj = build_adj(args, train_df, device)

    base_model = SimpleGNN(
        n_nodes=n_nodes,
        window_size=int(args.window_size),
        n_classes=n_classes,
        hidden_dim=int(args.hidden_dim),
        graph_type=args.graph_type,
        dropout=float(args.dropout),
        alpha=float(args.alpha),
    )

    model = SimpleGNNWithAdj(
        model=base_model,
        adj=adj,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    train_start_time = time.perf_counter()

    train_gnn(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=str(save_dir),
        save_best=True,
        average=args.average,
        # patience=args.patience,
    )

    train_total_time = time.perf_counter() - train_start_time
    avg_epoch_time = train_total_time / args.epochs

    print(f"Total training time: {train_total_time:.2f} sec")
    print(f"Average epoch time: {avg_epoch_time:.2f} sec")

    timing_metrics = {
        "total_training_time_sec": train_total_time,
        "average_epoch_time_sec": avg_epoch_time,
        "epochs": args.epochs,
    }

    with open(save_dir / "training_time.json", "w", encoding="utf-8") as f:
        json.dump(timing_metrics, f, ensure_ascii=False, indent=2)

    if args.eval_test:
        print("Evaluating test set using last-epoch model...")
        evaluate_and_save(
            model=model,
            dataset=test_ds,
            batch_size=args.batch_size,
            device=device,
            average=args.average,
            save_dir=save_dir,
            split_name="test_last_epoch",
        )

    best_ckpt_path = save_dir / "best_model.pt"

    if not best_ckpt_path.exists():
        print("best_model.pt not found, skipping best-model post-training evaluation.")
        return

    print(f"Loading best model from: {best_ckpt_path}")
    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.eval_train:
        print("Evaluating train set using best model...")
        evaluate_and_save(
            model=model,
            dataset=train_ds,
            batch_size=args.batch_size,
            device=device,
            average=args.average,
            save_dir=save_dir,
            split_name="train",
        )

    if args.eval_test:
        print("Evaluating test set using best model...")
        evaluate_and_save(
            model=model,
            dataset=test_ds,
            batch_size=args.batch_size,
            device=device,
            average=args.average,
            save_dir=save_dir,
            split_name="test",
        )


if __name__ == "__main__":
    main()