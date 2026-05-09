#!/bin/bash
set -e

# =========================================================================
# 1. 实验全局配置 (CNN)
# =========================================================================
DATASETS=("busi")
CNN_MODELS=("UNet" "AttUNet" "UNetplus" "UNet3plus" "UNeXt" "CMUNet" "CMUNeXt" "UKAN")

CONFIG_FILE="config.json"
NUM_FOLDS=5

# 🚀 划分 Results 文件夹
RESULTS_DIR="results/cnn"
mkdir -p "$RESULTS_DIR"

# =========================================================================
# 2. 辅助 Python 脚本
# =========================================================================
cat << 'EOF' > modify_cnn_config.py
import sys, json

config_file, exp_name, arch_type = sys.argv[1], sys.argv[2], sys.argv[3]

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

config['name'] = exp_name
config['arch']['type'] = arch_type

# 🚀 划分 Checkpoints 文件夹
config['trainer']['checkpoint_dir'] = f"checkpoints/cnn/{exp_name}"

with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
EOF

# =========================================================================
# 3. 主循环入口
# =========================================================================
for dataset in "${DATASETS[@]}"; do
  for model in "${CNN_MODELS[@]}"; do

    EXP_NAME="${dataset}_cnn_${model}"
    echo -e "\n\033[1;36m=======================================================\033[0m"
    echo -e "\033[1;36m 🚀 STARTING CNN EXPERIMENT: [${EXP_NAME}]\033[0m"
    echo -e "\033[1;36m=======================================================\033[0m"

    python modify_cnn_config.py "$CONFIG_FILE" "$EXP_NAME" "$model"

    CHECKPOINT_BASE_DIR=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['trainer']['checkpoint_dir'])")
    CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR%/}

    RESULTS_CSV="${RESULTS_DIR}/results_${EXP_NAME}.csv"
    if [ ! -f "$RESULTS_CSV" ]; then
        echo "Fold,PA,DSC,HD95,IoU,GFLOPs,Params" > "$RESULTS_CSV"
    fi

    # 训练检查
    NEED_TRAIN=false
    for fold in $(seq 1 ${NUM_FOLDS}); do
        if [ ! -f "${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth" ]; then
            NEED_TRAIN=true
            break
        fi
    done

    if [ "$NEED_TRAIN" = true ]; then
      echo "🔥 开始 CNN 训练 (1-${NUM_FOLDS}折)..."
      python train.py -c "$CONFIG_FILE"
    else
      echo "✅ 所有权重已存在，跳过训练。"
    fi

    # 测试环节
    echo -e "\n--- 🧪 测试提取指标 ---"
    for fold in $(seq 1 ${NUM_FOLDS}); do
      CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"
      if [ -f "$CHECKPOINT_PATH" ]; then
        TEST_OUTPUT=$(python test.py -r "$CHECKPOINT_PATH" -c "$CONFIG_FILE" || true)

        PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs || echo "0")
        DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs || echo "0")
        HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs || echo "0")
        IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs || echo "0")
        GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
        PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs || echo "0")

        echo "${fold},${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> "$RESULTS_CSV"
        echo "📊 Fold ${fold} - DSC: ${DSC} | IoU: ${IOU}"
      else
        echo "❌ 警告: 找不到权重文件 ${CHECKPOINT_PATH}!"
      fi
    done
  done
done
echo -e "\n\033[1;36m🎉 ALL CNN EXPERIMENTS COMPLETED 🎉\033[0m"