#!/bin/bash

# =========================================================================
# USFM + SegViT (官方神装模式) 三策略自动化脚本
# 包含：Fully Tuning, Freeze Encoder, From Scratch
# =========================================================================

set -e

# 核心配置
DATASET="busi"
CONFIG_FILE="config_official.json"  # 必须使用官方神装配置文件
NUM_FOLDS=5
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# 官方预训练权重路径 (请根据实际情况确认)
PRETRAIN_PATH="./pretrained_models/USFM_latest.pth"

# =========================================================================
# 辅助 Python 脚本：安全修改 JSON，切换策略并分配独立权重文件夹
# =========================================================================
cat << 'EOF' > modify_official_config.py
import sys, json

config_file = sys.argv[1]
exp_name = sys.argv[2]
freeze_mode = sys.argv[3]
pretrain_ckpt = sys.argv[4]

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

# 强制注入实验名称、冻结模式和预训练权重路径
config['name'] = exp_name
config['trainer']['freeze_mode'] = freeze_mode
config['usfm_args']['PRETRAIN_CKPT'] = pretrain_ckpt

# 🌟 新增：为每个策略指定一个独立的子文件夹来存放权重
config['trainer']['checkpoint_dir'] = f"checkpoints_official/{exp_name}"

# 确保使用的是 SegViT 官方模式
config['usfm_args']['mode'] = 'official'
config['usfm_args']['decoder_type'] = 'SegViT'
config['arch']['type'] = 'USFM_SegmentationModel'

with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
EOF

# =========================================================================
# 核心执行函数
# =========================================================================
run_strategy() {
    local STRATEGY_NAME=$1
    local FREEZE_MODE=$2
    local PRETRAIN_CKPT=$3

    local EXP_NAME="${DATASET}_usfm_segvit_official_${STRATEGY_NAME}"
    echo -e "\n======================================================="
    echo " 🚀 开始执行策略: [${STRATEGY_NAME}] "
    echo " 实验名称: ${EXP_NAME} | Freeze: ${FREEZE_MODE} | Pretrain: ${PRETRAIN_CKPT}"
    echo "======================================================="

    # 1. 动态修改 config_official.json (包含文件夹划分)
    python modify_official_config.py "$CONFIG_FILE" "$EXP_NAME" "$FREEZE_MODE" "$PRETRAIN_CKPT"

    # 2. 从被修改后的 JSON 中读取当前策略专属的 checkpoint_dir
    CHECKPOINT_BASE_DIR=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['trainer']['checkpoint_dir'])")
    CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR%/} # 去除尾部斜杠

    local RESULTS_CSV="${RESULTS_DIR}/results_${EXP_NAME}.csv"
    if [ ! -f "$RESULTS_CSV" ]; then
        echo "Fold,PA,DSC,HD95,IoU,GFLOPs,Params" > "$RESULTS_CSV"
    fi

    # 3. 检查是否需要训练
    local NEED_TRAIN=false
    for fold in $(seq 1 ${NUM_FOLDS}); do
        CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"
        if [ ! -f "$CHECKPOINT_PATH" ]; then
            NEED_TRAIN=true
            break
        fi
    done

    if [ "$NEED_TRAIN" = true ]; then
        echo "🔥 发现缺失权重，开始执行全量训练 (将自动训练 1-${NUM_FOLDS} 折)..."
        # 调用一次 train.py 进行全折训练
        python train.py -c "$CONFIG_FILE"
    else
        echo "✅ 所有权重均已存在，跳过训练，直接进入测试环节..."
    fi

    # 4. 按折进行测试并提取指标
    for fold in $(seq 1 ${NUM_FOLDS}); do
        CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"

        echo -e "\n--- 测试 Fold ${fold} ---"
        if [ -f "$CHECKPOINT_PATH" ]; then
            echo "🧪 正在提取指标，使用的权重: ${CHECKPOINT_PATH}"
            TEST_OUTPUT=$(python test.py -r "$CHECKPOINT_PATH" || true)

            PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs || echo "0")
            DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs || echo "0")
            HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs || echo "0")
            IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs || echo "0")
            GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
            PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs || echo "0")

            echo "${fold},${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> "$RESULTS_CSV"
            echo "📊 Fold ${fold} 结果: DSC=${DSC}, IoU=${IOU}"
        else
            echo "❌ 警告: 训练完成后仍未找到权重 ${CHECKPOINT_PATH}，请检查是否出现异常!"
        fi
    done
    echo "🏁 策略 [${STRATEGY_NAME}] 运行结束。"
}

# =========================================================================
# 执行三大策略
# =========================================================================

run_strategy "fully_tuning" "none" "$PRETRAIN_PATH"
run_strategy "freeze_encoder" "encoder" "$PRETRAIN_PATH"
run_strategy "from_scratch" "none" "none"

echo -e "\n🎉 所有 USFM-SegViT 策略测试执行完毕! 结果已存入 ${RESULTS_DIR}/ 目录下。"