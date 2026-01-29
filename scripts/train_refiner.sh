#!/bin/bash

DATA_DIR="data/processed"
EXP_NAME="obj_weight_5"
HIGH_RES=128
LATENT_RES=32
K_STEP=1
BATCH_SIZE=16
EPOCHS=500
VIS_INTERVAL=5
LR=1e-5
SAVE_INTERVAL=5

# Model Config
MODEL_CHANNELS=128
CHANNEL_MULT="1 2 4 8"
NUM_RES_BLOCKS=2
OBJ_WEIGHT=1.0

SUBSET_N="1000"

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -eq "0" ]; then
    NUM_GPUS=1
fi

echo "Running on $NUM_GPUS GPUs"

PORT=29500
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT scripts/train.py \
    --data_dir "$DATA_DIR" \
    --exp_name "$EXP_NAME" \
    --high_res $HIGH_RES \
    --latent_res $LATENT_RES \
    --k_step $K_STEP \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --vis_interval $VIS_INTERVAL \
    --lr $LR \
    --model_channels $MODEL_CHANNELS \
    --channel_mult $CHANNEL_MULT \
    --num_res_blocks $NUM_RES_BLOCKS \
    --obj_weight $OBJ_WEIGHT \
    --subset_n "$SUBSET_N" \
    --num_workers 8 \
    --save_interval $SAVE_INTERVAL
