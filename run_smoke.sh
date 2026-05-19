#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-SDAHU}"
DATA_ROOT="${DATA_ROOT:-/workspace/LBNL_FDD/data/processed}"
SAVE_ROOT="${SAVE_ROOT:-outputs/smoke}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
STRIDE="${STRIDE:-1}"

mkdir -p logs "${SAVE_ROOT}"

echo "PWD: $(pwd)"
python --version
nvidia-smi || true

python scripts/train_mlp.py \
  --dataset "$DATASET" \
  --data_root "$DATA_ROOT" \
  --save_root "$SAVE_ROOT" \
  --window_size "$WINDOW_SIZE" \
  --stride "$STRIDE" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --standardize \
  --eval_train \
  --eval_test \
  --epochs 1 \
  --run_name "smoke_mlp_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_hidden 128 \
  --n_layers 2 \
  --dropout 0.3 2>&1 | tee logs/smoke_mlp.log

echo "Smoke test finished"