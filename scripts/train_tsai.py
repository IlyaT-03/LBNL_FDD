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
from lbnl_fdd.training.train_tsai import train_tsai
from lbnl_fdd.training.evaluate_tsai import eval_tsai


def build_model(args, n_features, n_classes):
    name = args.model

    if name == "lstm_fcn":
        from tsai.models.RNN_FCN import MLSTM_FCN
        return MLSTM_FCN(
            c_in=n_features,
            c_out=n_classes,
            seq_len=args.window_size,
            hidden_size=args.hidden_size,
            rnn_layers=args.rnn_layers,
            cell_dropout=args.cell_dropout,
            rnn_dropout=args.rnn_dropout,
            fc_dropout=args.fc_dropout,
        )

    if name == "lstm":
        from tsai.models.RNNPlus import LSTMPlus
        return LSTMPlus(c_in=n_features, c_out=n_classes)

    if name == "gru":
        from tsai.models.RNNPlus import GRUPlus
        return GRUPlus(c_in=n_features, c_out=n_classes)

    if name == "inceptiontime":
        from tsai.models.InceptionTime import InceptionTime
        return InceptionTime(c_in=n_features, c_out=n_classes)

    if name == "resnet":
        from tsai.models.ResNet import ResNet
        return ResNet(c_in=n_features, c_out=n_classes)

    if name == "fcn":
        from tsai.models.FCN import FCN
        return FCN(c_in=n_features, c_out=n_classes)

    raise ValueError(f"Unknown model: {name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train tsai model")

    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm_fcn", "lstm", "gru",
                                 "inceptiontime", "resnet", "fcn"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--save_root", type=str, default="outputs/runs")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # lstm_fcn гиперпараметры
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--cell_dropout", type=float, default=0.1)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--fc_dropout", type=float, default=0.0)

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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(data_dir, split):
    df = pd.read_csv(data_dir / f"{split}_df.csv", index_col=[0, 1])
    target = pd.read_csv(
        data_dir / f"{split}_target.csv", index_col=[0, 1]
    ).iloc[:, 0]
    return df, target


def build_dataset(df, target, window_size, stride, windows_file, split_name):
    if windows_file is None:
        return SlidingWindowDataset(
            df=df, target=target,
            window_size=window_size, stride=stride,
        )
    print(f"Using selected {split_name} windows: {windows_file}")
    windows_df = pd.read_json(windows_file, lines=True)
    return SelectedWindowsDataset(df=df, windows_df=windows_df)


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = args.run_name or f"{args.model}_run"
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    data_dir = Path(args.data_root) / args.dataset
    save_dir = Path(args.save_root) / args.dataset / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Model: {args.model} | Device: {device}")

    train_df, train_target = load_split(data_dir, "train")
    val_df,   val_target   = load_split(data_dir, "val")
    test_df,  test_target  = load_split(data_dir, "test")

    if args.standardize:
        scaler = StandardScaler()
        train_df = pd.DataFrame(scaler.fit_transform(train_df),
                                index=train_df.index, columns=train_df.columns)
        val_df   = pd.DataFrame(scaler.transform(val_df),
                                index=val_df.index,   columns=val_df.columns)
        test_df  = pd.DataFrame(scaler.transform(test_df),
                                index=test_df.index,  columns=test_df.columns)

    train_ds = build_dataset(train_df, train_target, args.window_size,
                             args.stride, args.train_windows_file, "train")
    val_ds   = build_dataset(val_df,   val_target,   args.window_size,
                             args.stride, args.val_windows_file,   "val")
    test_ds  = build_dataset(test_df,  test_target,  args.window_size,
                             args.stride, args.test_windows_file,  "test")

    n_features = int(train_df.shape[1])
    n_classes  = int(train_target.nunique())

    print(f"Features: {n_features} | Classes: {n_classes}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    model = build_model(args, n_features, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    train_start = time.perf_counter()

    train_tsai(
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

    train_time = time.perf_counter() - train_start
    print(f"Training time: {train_time:.2f}s | Avg epoch: {train_time / args.epochs:.2f}s")

    with open(save_dir / "training_time.json", "w") as f:
        json.dump({"total": train_time, "per_epoch": train_time / args.epochs,
                   "epochs": args.epochs}, f, indent=2)

    best_ckpt = save_dir / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    def run_eval(ds, split_name):
        metrics, y_true, y_pred = eval_tsai(
            model=model, window_ds=ds,
            batch_size=args.batch_size, device=device,
            average=args.average,
        )
        print(f"{split_name}: {metrics}")
        with open(save_dir / f"{split_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        np.save(save_dir / f"{split_name}_y_true.npy", y_true)
        np.save(save_dir / f"{split_name}_y_pred.npy", y_pred)

    if args.eval_train:
        run_eval(train_ds, "train")
    if args.eval_test:
        run_eval(test_ds, "test")


if __name__ == "__main__":
    main()