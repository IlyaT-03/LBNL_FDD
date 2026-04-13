import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from lbnl_fdd.data.sliding_window import SlidingWindowDataset
from lbnl_fdd.data.selected_window import SelectedWindowsDataset
from lbnl_fdd.models.gnn.gnn import GNN_TAM
from lbnl_fdd.training.train_gnn import train_gnn


def parse_args():
    parser = argparse.ArgumentParser(description="Train GNN_TAM")

    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. SDAHU")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--save_root", type=str, default="outputs/runs")
    parser.add_argument("--run_name", type=str, default="gnn_tam_run1")

    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)

    parser.add_argument("--n_gnn", type=int, default=1)
    parser.add_argument("--gsl_type", type=str, default="relu")
    parser.add_argument("--n_hidden", type=int, default=1024)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--average", type=str, default="macro")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--standardize", action="store_true")

    parser.add_argument("--train_windows_file", type=str, default=None)
    parser.add_argument("--val_windows_file", type=str, default=None)

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

    if args.train_windows_file is None:
        train_ds = SlidingWindowDataset(
            df=train_df,
            target=train_target,
            window_size=args.window_size,
            stride=args.stride,
        )
    else:
        print(f"Using selected train windows: {args.train_windows_file}")
        train_windows_df = pd.read_json(args.train_windows_file, lines=True)
        train_ds = SelectedWindowsDataset(
            df=train_df,
            windows_df=train_windows_df,
        )

    if args.val_windows_file is None:
        val_ds = SlidingWindowDataset(
            df=val_df,
            target=val_target,
            window_size=args.window_size,
            stride=args.stride,
        )
    else:
        print(f"Using selected val windows: {args.val_windows_file}")
        val_windows_df = pd.read_json(args.val_windows_file, lines=True)
        val_ds = SelectedWindowsDataset(
            df=val_df,
            windows_df=val_windows_df,
        )

    print(f"Train windows: {len(train_ds)}")
    print(f"Val windows: {len(val_ds)}")
    print(f"Num features: {train_df.shape[1]}")
    print(f"Num classes: {train_target.nunique()}")

    model = GNN_TAM(
        n_nodes=int(train_df.shape[1]),
        window_size=int(args.window_size),
        n_classes=int(train_target.nunique()),
        n_gnn=int(args.n_gnn),
        gsl_type=args.gsl_type,
        n_hidden=int(args.n_hidden),
        alpha=float(args.alpha),
        k=args.k,
        device=device,
    )

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
    )


if __name__ == "__main__":
    main()