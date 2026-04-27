#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- CNN Experiment Configuration ---
#DATASETS=("busi" "bus_uc" "busbra" "bus_uclm" "yap2018")
DATASETS=("busi")
CNN_MODELS=("UNet")
# CNN_MODELS=("UNet" "AttUNet" "UNetplus" "UNet3plus" "UNeXt" "CMUNet" "CMUNeXt")

# 优化：直接定义为字符串，方便传入 Python 解析
CONFIG_FILE="config.json"
NUM_FOLDS=5

# ==========================================
# 🎯 核心优化：让 Shell 脚本自动提取 JSON 里的 Checkpoint 路径
# ==========================================
CHECKPOINT_BASE_DIR=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['trainer']['checkpoint_dir'])")
# 去除可能存在的尾部斜杠，保证路径拼接安全
CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR%/}

echo "Detected Checkpoint Directory from JSON: ${CHECKPOINT_BASE_DIR}"

# --- Main Experiment Loop for CNNs ---
for dataset in "${DATASETS[@]}"; do
  for model in "${CNN_MODELS[@]}"; do

    echo -e "\n\n======================================================="
    echo "  STARTING CNN EXPERIMENT: DATASET=${dataset} | MODEL=${model}"
    echo "======================================================="

    RUN_ID="lovasz_${dataset}_${model}"

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
      python train.py -c ${CONFIG_FILE} \
                      --name "${RUN_ID}" \
                      --datasets "${dataset}" \
                      --model ${model}
    fi

    # Testing & Aggregation Phase
    echo -e "\n--- Testing and Aggregating Results for ${RUN_ID} ---"

    RESULTS_DIR="results"
    mkdir -p ${RESULTS_DIR}
    # 结果文件名动态附加 checkpoint 文件夹名称，防止覆盖旧实验结果
    RESULTS_CSV="${RESULTS_DIR}/lovasz_results_${RUN_ID}_$(basename ${CHECKPOINT_BASE_DIR}).csv"

    if [ ! -f "$RESULTS_CSV" ]; then
        echo "PA,DSC,HD95,IoU,GFLOPs,Params" > ${RESULTS_CSV}
    fi

    for fold in $(seq 1 ${NUM_FOLDS}); do
      CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${RUN_ID}_fold${fold}_best.pth"

      if [ -f "$CHECKPOINT_PATH" ]; then
        # 加入 || true 防止测试出错时整个流水线崩溃
        TEST_OUTPUT=$(python test.py -r "$CHECKPOINT_PATH" || true)

        # 加入 || echo "0" 防止 grep 匹配失败导致 xargs 报错
        PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs || echo "0")
        DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs || echo "0")
        HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs || echo "0")
        IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs || echo "0")
        GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
        PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs || echo "0")

        echo "${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> ${RESULTS_CSV}
      else
        echo "Warning: Checkpoint for fold ${fold} not found at ${CHECKPOINT_PATH}!"
      fi
    done

    echo "--- CNN EXPERIMENT FINISHED: ${RUN_ID} ---"
  done
done

echo -e "\n\n======================================================="
echo "  ALL CNN EXPERIMENTS COMPLETED"
echo "======================================================="