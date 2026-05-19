#!/usr/bin/env bash
set -euo pipefail
pip install -e /workspace/LBNL_FDD/ -q
python -c "import inspect; from lbnl_fdd.models.gru import GRUClassifier; print(inspect.signature(GRUClassifier.__init__))"

DATASET="${DATASET:-SDAHU}"
DATA_ROOT="${DATA_ROOT:-/workspace/LBNL_FDD/data/processed}"
SAVE_ROOT="${SAVE_ROOT:-outputs/runs/various_seeds}"
DEVICE="${DEVICE:-cuda}"
WINDOW_SIZE="${WINDOW_SIZE:-100}"
STRIDE="${STRIDE:-1}"
TIMESNET_EPOCHS="${TIMESNET_EPOCHS:-20}"

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

for SEED in 1 2 3; do
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

  run_cmd "gru_${DATASET}_seed${SEED}" \
    python scripts/train_gru.py \
    "${COMMON_ARGS[@]}" \
    --epochs "$TIMESNET_EPOCHS" \
    --run_name "seeds_gru_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
    --batch_size 512 \
    --lr 1e-3 \
    --hidden_dim 128 \
    --num_layers 2 \
    --dropout 0.1 \
    --concat_layers 1

  run_cmd "lstm_${DATASET}_seed${SEED}" \
  python scripts/train_lstm.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "seeds_lstm_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --hidden_dim 128 \
  --num_layers 2 \
  --dropout 0.1 \

  run_cmd "mlp_m_${DATASET}_seed${SEED}" \
  python scripts/train_mlp.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$MLP_EPOCHS" \
  --run_name "seeds_mlp_m_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --n_hidden 512 \
  --n_layers 2 \
  --dropout 0.3


    run_cmd "timesnet_${DATASET}_seed${SEED}" \
      python scripts/train_timesnet.py \
      "${COMMON_ARGS[@]}" \
      --epochs "$TIMESNET_EPOCHS" \
      --run_name "seeds_timesnet_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
      --batch_size 512 \
      --lr 1e-3 \
      --d_model 64 \
      --d_ff 128 \
      --e_layers 2 \
      --top_k 3 \
      --num_kernels 4 \
      --dropout 0.1

    run_cmd "nonstationary_transformer_${DATASET}_seed${SEED}" \
  python scripts/train_nonstationary_transformer.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "seed_nonstationary_transformer_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
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

  run_cmd "informer_${DATASET}_seed${SEED}" \
  python scripts/train_informer.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "seeds_informer_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --d_model 128 \
  --d_ff 256 \
  --e_layers 2 \
  --n_heads 4 \
  --factor 5 \
  --dropout 0.1 \
  --activation gelu

  run_cmd "cnn1d_${DATASET}_seed${SEED}" \
  python scripts/train_1dcnn.py \
  "${COMMON_ARGS[@]}" \
  --epochs "$TIMESNET_EPOCHS" \
  --run_name "seeds_cnn1d_w${WINDOW_SIZE}_s${STRIDE}_seed${SEED}" \
  --batch_size 512 \
  --lr 1e-3 \
  --conv1_multiplier 4 \
  --conv2_multiplier 16 \
  --kernel_size 5 \
  --conv_stride 5 \
  --pool_size 2 \
  --pool_stride 2 \
  --hidden_dim 256 \
  --dropout 0.0
done

echo "Done. Results saved under: ${SAVE_ROOT}/${DATASET}"
echo "Logs saved under: logs/"