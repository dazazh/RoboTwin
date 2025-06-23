#!/bin/bash

# 检查参数数量
if [ $# -ne 4 ]; then
    echo "Usage: ./run_all.sh {task_name} {camera_type} {episode_num} {gpu_id}"
    exit 1
fi

# 解析参数
TASK_NAME=$1
CAMERA_TYPE=$2
EPISODE_NUM=$3
GPU_ID=$4

# 0. 删除历史数据
rm -rf data/${TASK_NAME}_${CAMERA_TYPE}_pkl/*

# 1. 生成数据
echo "Step 1: Generating data..."
bash run_task.sh ${TASK_NAME} ${GPU_ID}

# 2. 转换格式
echo "Step 2: Converting pkl to zarr..."
python script/pkl2zarr_dp.py ${TASK_NAME} ${CAMERA_TYPE} ${EPISODE_NUM}

# 3. 进入训练目录
echo "Step 3: Changing to policy/My-Policy..."
cd policy/My-Policy || { echo "Directory not found: policy/My-Policy"; exit 1; }

# 4. 执行训练
echo "Step 4: Training model..."
bash train.sh ${TASK_NAME} ${CAMERA_TYPE} ${EPISODE_NUM} 0 ${GPU_ID}

# 5. 执行评估
echo "Step 5: Evaluating model..."
bash eval.sh ${TASK_NAME} ${CAMERA_TYPE} ${EPISODE_NUM} 300 0 ${GPU_ID}

echo "✅ All steps completed successfully!"

# example: bash train_eval_dp_RoboTwinBench.sh empty_cup_place L515 100 0``