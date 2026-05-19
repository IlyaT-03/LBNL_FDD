#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-SDAHU}"
DATA_ROOT="${DATA_ROOT:-/workspace/LBNL_FDD/data/processed}"
SAVE_ROOT="${SAVE_ROOT:-outputs/runs}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
STRIDE="${STRIDE:-1}"

MLP_EPOCHS="${MLP_EPOCHS:-20}"
TIMESNET_EPOCHS="${TIMESNET_EPOCHS:-20}"
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

run_cmd "cnn1d_${DATASET}" \
  python scripts/train_1dcnn.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "cnn1d_larger_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --conv1_multiplier 4 \
  --conv2_multiplier 16 \
  --kernel_size 5 \
  --conv_stride 5 \
  --pool_size 2 \
  --pool_stride 2 \
  --hidden_dim 2048 \
  --dropout 0.0

echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"