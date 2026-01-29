#!/bin/bash

cd "$(dirname "$0")" || exit 1
CKPT="${CKPT:-./experiments/<exp_name>/checkpoints/model_epoch_0001.pth}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
OUT_DIR="${OUT_DIR:-./results/<exp_name>}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMP="${TEMP:-0.1}"

python inference.py \
  --checkpoint "$CKPT" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --temp "$TEMP"