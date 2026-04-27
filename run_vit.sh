#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- ViT Experiment Configuration ---
DATASETS=("busi")
VIT_MODELS=("USFM_UPerNet")
# VIT_MODELS=("TransUnet" "JEPA_UPerNet" "SwinUnet" "MedT" "USFM_UPerNet")
CONFIG_FILE="config.json"
NUM_FOLDS=5


CHECKPOINT_BASE_DIR=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['trainer']['checkpoint_dir'])")
# 去除可能存在的尾部斜杠
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR%/}

echo "Detected Checkpoint Directory from JSON: ${CHECKPOINT_BASE_DIR}"

# --- Main Experiment Loop for ViTs ---
for dataset in "${DATASETS[@]}"; do
  for model in "${VIT_MODELS[@]}"; do

    echo -e "\n\n======================================================="
    echo "  STARTING ViT EXPERIMENT: DATASET=${dataset} | MODEL=${model}"
    echo "======================================================="

    RUN_ID="${dataset}_${model}"

    # Check if training is already completed
    NUM_CHECKPOINTS=0
    if [ -d "$CHECKPOINT_BASE_DIR" ]; then
        NUM_CHECKPOINTS=$(find "$CHECKPOINT_BASE_DIR" -name "${RUN_ID}_fold*_best.pth" | wc -l)
    fi

    # Training Phase (Conditional)
    if [ "$NUM_CHECKPOINTS" -eq "$NUM_FOLDS" ]; then
      echo "All ${NUM_FOLDS} checkpoints found for ${RUN_ID}. Skipping training."
    else
      echo "Starting training for ${RUN_ID}. Found ${NUM_CHECKPOINTS}/${NUM_FOLDS} completed folds."
      # 注意：去掉了 --name 参数，因为 train.py 内部本来就会从 json 的 name 生成 run_name
      python train.py -c ${CONFIG_FILE} \
                      --datasets "${dataset}" \
                      --model ${model}
    fi

    # Testing & Aggregation Phase
    echo -e "\n--- Testing and Aggregating Results for ${RUN_ID} ---"

    RESULTS_DIR="results"
    mkdir -p ${RESULTS_DIR}
    # 结果文件名最好也能带上 checkpoint 文件夹的名字，方便区分不同实验
    RESULTS_CSV="${RESULTS_DIR}/results_${RUN_ID}_$(basename ${CHECKPOINT_BASE_DIR}).csv"

    # 只有当文件不存在时才写入表头
    if [ ! -f "$RESULTS_CSV" ]; then
        echo "PA,DSC,HD95,IoU,GFLOPs,Params" > ${RESULTS_CSV}
    fi

    for fold in $(seq 1 ${NUM_FOLDS}); do
      CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${RUN_ID}_fold${fold}_best.pth"

      if [ -f "$CHECKPOINT_PATH" ]; then
        # 提取指标日志 (使用 || true 防止 grep 失败导致脚本中断)
        TEST_OUTPUT=$(python test.py -r "$CHECKPOINT_PATH" || true)

        PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs || echo "0")
        DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs || echo "0")
        HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs || echo "0")
        IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs || echo "0")
        GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
        PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs || echo "0")

        echo "${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> ${RESULTS_CSV}
      else
        echo "Warning: Checkpoint for fold ${fold} not found at ${CHECKPOINT_PATH} (Testing skipped for this fold)."
      fi
    done

    echo "--- ViT EXPERIMENT FINISHED: ${RUN_ID} ---"
  done
done

echo -e "\n\n======================================================="
echo "  ALL ViT EXPERIMENTS COMPLETED"
echo "======================================================="