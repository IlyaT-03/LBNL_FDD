#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-SDAHU}"
DATA_ROOT="${DATA_ROOT:-data/preprocessed_data}"
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

# ---------------------------
# MLP family
# ---------------------------
run_cmd "mlp_s_${DATASET}" \
  python scripts/train_mlp.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$MLP_EPOCHS" \
  --run_name "mlp_s_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_hidden 128 \
  --n_layers 2 \
  --dropout 0.3

run_cmd "mlp_m_${DATASET}" \
  python scripts/train_mlp.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$MLP_EPOCHS" \
  --run_name "mlp_m_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_hidden 512 \
  --n_layers 2 \
  --dropout 0.3

run_cmd "mlp_l_${DATASET}" \
  python scripts/train_mlp.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$MLP_EPOCHS" \
  --run_name "mlp_l_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_hidden 1024 \
  --n_layers 2 \
  --dropout 0.3

# ---------------------------
# Sequence models
# ---------------------------
run_cmd "timesnet_${DATASET}" \
  python scripts/train_timesnet.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "timesnet_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 16 \
  --lr 1e-3 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --top_k 3 \
  --num_kernels 4 \
  --dropout 0.1

run_cmd "cnn1d_${DATASET}" \
  python scripts/train_cnn1d.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "cnn1d_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 16 \
  --lr 1e-3 \
  --conv1_multiplier 4 \
  --conv2_multiplier 16 \
  --kernel_size 5 \
  --conv_stride 5 \
  --pool_size 2 \
  --pool_stride 2 \
  --hidden_dim 256 \
  --dropout 0.0

run_cmd "informer_${DATASET}" \
  python scripts/train_informer.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "informer_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 16 \
  --lr 1e-3 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 4 \
  --factor 5 \
  --dropout 0.1 \
  --activation gelu

run_cmd "gru_${DATASET}" \
  python scripts/train_gru.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "gru_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 16 \
  --lr 1e-3 \
  --hidden_dim 128 \
  --n_layers 2 \
  --dropout 0.1

run_cmd "nonstationary_transformer_${DATASET}" \
  python scripts/train_nonstationary_transformer.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "nonstationary_transformer_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 16 \
  --lr 1e-3 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 4 \
  --factor 5 \
  --dropout 0.1 \
  --activation gelu \
  --p_hidden_dim 128 \
  --p_hidden_layers 2

# ---------------------------
# Simple GNN family
# ---------------------------
for GRAPH_TYPE in corr knn attention full; do
  run_cmd "simple_gnn_${GRAPH_TYPE}_${DATASET}" \
    python scripts/train_simple_gnn.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$GNN_EPOCHS" \
    --run_name "simple_gnn_${GRAPH_TYPE}_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
    --batch_size 512 \
    --lr 1e-3 \
    --graph_type "$GRAPH_TYPE" \
    --k 5 \
    --threshold 0.3 \
    --hidden_dim 1024 \
    --dropout 0.0 \
    --alpha 0.1
done

# ---------------------------
# GNN-TAM
# ---------------------------
run_cmd "gnn_tam_${DATASET}" \
  python scripts/train_gnn_tam.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$GNN_EPOCHS" \
  --run_name "gnn_tam_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_hidden 1024 \
  --n_gnn 1 \
  --gsl_type relu \
  --alpha 0.1

echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"