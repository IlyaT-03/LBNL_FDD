#!/usr/bin/env bash
set -euo pipefail

pip install -e /workspace/LBNL_FDD/ -q
python -c "import inspect; from lbnl_fdd.models.gru import GRUClassifier; print(inspect.signature(GRUClassifier.__init__))"

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



run_cmd "gru_${DATASET}" \
  python scripts/train_gru.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "gru_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}_concat_layers_fixed_lr_smaller_50_epochs" \
  --batch_size 512 \
  --lr 1e-4 \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.1 \
  --concat_layers 1



echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"