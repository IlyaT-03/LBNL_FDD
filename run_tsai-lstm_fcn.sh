#!/usr/bin/env bash
set -euo pipefail

pip install tsai -q


DATASET="${DATASET:-SDAHU}"
DATA_ROOT="${DATA_ROOT:-/workspace/LBNL_FDD/data/processed}"
SAVE_ROOT="${SAVE_ROOT:-outputs/runs}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
STRIDE="${STRIDE:-100}"

TSAI_EPOCHS="${TSAI_EPOCHS:-20}"

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

# ---------------------------
# tsai models — smoke test
# ---------------------------
for MODEL in lstm_fcn; do
  run_cmd "${MODEL}_${DATASET}" \
    python scripts/train_tsai.py \
    "${COMMON_ARGS[@]}" \
    --model "$MODEL" \
    --epochs "$TSAI_EPOCHS" \
    --run_name "${MODEL}_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_lstm_fcn_fixed" \
    --batch_size 512 \
    --lr 1e-3
    --hidden_size 128 \
    --rnn_layers 2 \
    --cell_dropout 0.1 \
    --rnn_dropout 0.1 \
    --fc_dropout 0.0
done

echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"