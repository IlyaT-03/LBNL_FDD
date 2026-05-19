#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-SDAHU}"
DATA_ROOT="${DATA_ROOT:-/workspace/LBNL_FDD/data/processed}"
SAVE_ROOT="${SAVE_ROOT:-outputs/runs}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
STRIDE="${STRIDE:-1}"

GNN_EPOCHS="${GNN_EPOCHS:-100}"

mkdir -p logs

run_cmd() {
  local name="$1"
  shift

  echo "=================================================="
  echo "Running: $name"
  echo "Command: $*"
  echo "=================================================="

  "$@" 2>&1 | tee "logs/${name}.log"
}

COMMON_ARGS=(
  --dataset "$DATASET"
  --data_root "$DATA_ROOT"
  --save_root "$SAVE_ROOT"
  --window_size "$WINDOW_SIZE"
  --stride "$STRIDE"
  --device "$DEVICE"
  --seed "$SEED"
  --standardize
  --eval_train
  --eval_test
)

# GNN-TAM baseline
run_cmd "gnn_tam_${DATASET}_base" \
  python scripts/train_gnn_tam.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$GNN_EPOCHS" \
  --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_base" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_gnn 1 \
  --gsl_type tanh \
  --n_hidden 1024 \
  --alpha 0.1

# n_hidden ablation
for H in 512 2048; do
  run_cmd "gnn_tam_${DATASET}_h${H}" \
    python scripts/train_gnn_tam.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$GNN_EPOCHS" \
    --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_h${H}" \
    --batch_size 512 \
    --lr 1e-3 \
    --n_gnn 1 \
    --gsl_type tanh \
    --n_hidden "$H" \
    --alpha 0.1
done

# n_gnn ablation
run_cmd "gnn_tam_${DATASET}_g2" \
  python scripts/train_gnn_tam.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$GNN_EPOCHS" \
  --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_g2" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_gnn 2 \
  --gsl_type tanh \
  --n_hidden 1024 \
  --alpha 0.1

# alpha ablation
for A in 0.05 0.2; do
  run_cmd "gnn_tam_${DATASET}_a${A}" \
    python scripts/train_gnn_tam.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$GNN_EPOCHS" \
    --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_a${A}" \
    --batch_size 512 \
    --lr 1e-3 \
    --n_gnn 1 \
    --gsl_type tanh \
    --n_hidden 1024 \
    --alpha "$A"
done

# k ablation
for K in 5 10; do
  run_cmd "gnn_tam_${DATASET}_k${K}" \
    python scripts/train_gnn_tam.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$GNN_EPOCHS" \
    --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_k${K}" \
    --batch_size 512 \
    --lr 1e-3 \
    --n_gnn 1 \
    --gsl_type tanh \
    --n_hidden 1024 \
    --alpha 0.1 \
    --k "$K"
done

echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"