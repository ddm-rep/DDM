#!/bin/bash

cd "$(dirname "$0")" || exit 1

MASTER_PORT="${MASTER_PORT:-12356}"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

NOTE=""
GPUS="${GPUS:-8}"
DATA_DIR="${DATA_DIR:-/workspace/data/processed_32}"
VIS_INTERVAL="${VIS_INTERVAL:-}"
VIS_FRAMES="${VIS_FRAMES:-}"
VIS_FPS="${VIS_FPS:-}"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --note) NOTE="$2"; shift ;;
        --gpus) GPUS="$2"; shift ;;
        --data_dir) DATA_DIR="$2"; shift ;;
        --vis_interval) VIS_INTERVAL="$2"; shift ;;
        --vis_frames) VIS_FRAMES="$2"; shift ;;
        --vis_fps) VIS_FPS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Running Planner Experiment: $NOTE on $GPUS GPUs"
echo "MASTER_PORT: $MASTER_PORT"

export OMP_NUM_THREADS=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1

CMD=(torchrun --nproc_per_node=$GPUS --master_port=$MASTER_PORT train_planner.py --note "$NOTE")
if [[ -n "$DATA_DIR" ]]; then
  CMD+=(--data_dir "$DATA_DIR")
fi
if [[ -n "$VIS_INTERVAL" ]]; then
  CMD+=(--vis_interval "$VIS_INTERVAL")
fi
if [[ -n "$VIS_FRAMES" ]]; then
  CMD+=(--vis_frames "$VIS_FRAMES")
fi
if [[ -n "$VIS_FPS" ]]; then
  CMD+=(--vis_fps "$VIS_FPS")
fi
"${CMD[@]}"
