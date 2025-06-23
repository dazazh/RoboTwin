#!/bin/bash

TASK_NAME=$1
CAMERA_TYPE=$2
NUM_EPISODES=$3
GPU_ID=$4

# 可选参数
PKL_ROOT="/data/user/xcs/yuhao/3d-aware/RoboTwin/data"
OUTPUT_DIR="./data"

echo "======== Running pkl2zarr_action_vggt.py ========"
echo "TASK_NAME: $TASK_NAME"
echo "CAMERA_TYPE: $CAMERA_TYPE"
echo "NUM_EPISODES: $NUM_EPISODES"
echo "GPU_ID: $GPU_ID"
echo "==============================================="

export CUDA_VISIBLE_DEVICES=$GPU_ID

python scripts/pkl2zarr_spatialdp.py \
    $TASK_NAME \
    $CAMERA_TYPE \
    $NUM_EPISODES \
    --pkl_root $PKL_ROOT \
    --output_dir $OUTPUT_DIR
