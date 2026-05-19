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

GSL_TYPES=(
  tanh
)

# ---------------------------
# GNN-TAM for all GSL types
# ---------------------------
for GSL_TYPE in "${GSL_TYPES[@]}"; do
  run_cmd "gnn_tam_${DATASET}_${GSL_TYPE}" \
    python scripts/train_gnn_tam.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$GNN_EPOCHS" \
    --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_gsl_${GSL_TYPE}" \
    --batch_size 512 \
    --lr 1e-3 \
    --n_hidden 1024 \
    --n_gnn 1 \
    --gsl_type "$GSL_TYPE" \
    --alpha 0.1
done

echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"