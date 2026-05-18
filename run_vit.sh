#!/bin/bash
set -e

# =========================================================================
# 1. 实验全局配置 (ViT)
# =========================================================================
DATASETS=("busi")
STANDARD_VITS=("TransUnet")
#STANDARD_VITS=("TransUnet" "SwinUnet" "MedT" "JEPA_UPerNet")
USFM_DECODERS=("UPerHead" "SegViT")
USFM_MODES=("local" "official")

CONFIG_FILE="config.json"
NUM_FOLDS=5
PRETRAIN_PATH="./pretrained_models/USFM_latest.pth"

# 🚀 划分 Results 文件夹
RESULTS_DIR="results/vit"
mkdir -p "$RESULTS_DIR"

# =========================================================================
# 2. 辅助 Python 脚本 (已修改：支持动态切换 Loss)
# =========================================================================
cat << 'EOF' > modify_vit_config.py
import sys, json

config_file = sys.argv[1]
exp_name = sys.argv[2]
arch_type = sys.argv[3]
decoder_type = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "none" else None
usfm_mode = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "none" else None
pretrain_ckpt = sys.argv[6] if len(sys.argv) > 6 else None

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

config['name'] = exp_name
config['arch']['type'] = arch_type
config['trainer']['checkpoint_dir'] = f"checkpoints/vit/{exp_name}"

# 🌟 核心修改：动态切换 Loss 函数
if decoder_type == 'SegViT':
    # 只要是 SegViT (无论是 local 还是 official)，都强制使用解析字典的 ATMLoss
    config['loss'] = {
        "type": "ATMLoss",
        "args": {
            "num_classes": 1,
            "dec_layers": 3,
            "mask_weight": 20.0,
            "dice_weight": 1.0,
            "cls_weight": 1.0
        }
    }
else:
    # ⚠️ 恢复为其他模型使用的常规 Loss。
    # 这里默认填了 BCEDiceLoss，如果你的 config.json 原本用的是 DiceLoss 或其他名字，请在这里修改！
    config['loss'] = {
        "type": "BCEDiceLoss",
        "args": {}
    }

if arch_type == 'USFM':
    if 'usfm_args' not in config:
        config['usfm_args'] = {}
    config['usfm_args']['decoder_type'] = decoder_type
    config['usfm_args']['mode'] = usfm_mode
    if pretrain_ckpt and pretrain_ckpt != "none":
        config['usfm_args']['PRETRAIN_CKPT'] = pretrain_ckpt
elif 'usfm_args' in config:
    config['usfm_args']['mode'] = 'none'

with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
EOF

# =========================================================================
# 3. 核心实验执行函数
# =========================================================================
run_experiment() {
    local EXP_NAME=$1
    local ARCH_TYPE=$2
    local DECODER_TYPE=$3
    local USFM_MODE=$4
    local PRETRAIN_CKPT=$5

    echo -e "\n\033[1;32m=======================================================\033[0m"
    echo -e "\033[1;32m 🚀 STARTING ViT EXPERIMENT: [${EXP_NAME}]\033[0m"
    if [ "$ARCH_TYPE" == "USFM" ]; then
        echo " ⚙️  ARCH: ${ARCH_TYPE} | Decoder: ${DECODER_TYPE} | Mode: ${USFM_MODE}"
    else
        echo " ⚙️  ARCH: ${ARCH_TYPE} (Standard ViT)"
    fi
    echo -e "\033[1;32m=======================================================\033[0m"

    python modify_vit_config.py "$CONFIG_FILE" "$EXP_NAME" "$ARCH_TYPE" "${DECODER_TYPE:-none}" "${USFM_MODE:-none}" "${PRETRAIN_CKPT:-none}"

    CHECKPOINT_BASE_DIR=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['trainer']['checkpoint_dir'])")
    CHECKPOINT_BASE_DIR=${CHECKPOINT_BASE_DIR%/}

    RESULTS_CSV="${RESULTS_DIR}/results_${EXP_NAME}.csv"
    if [ ! -f "$RESULTS_CSV" ]; then
        echo "Fold,PA,DSC,HD95,IoU,GFLOPs,Params" > "$RESULTS_CSV"
    fi

    NEED_TRAIN=false
    for fold in $(seq 1 ${NUM_FOLDS}); do
        if [ ! -f "${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth" ]; then
            NEED_TRAIN=true
            break
        fi
    done

    if [ "$NEED_TRAIN" = true ]; then
      echo "🔥 开始 ViT 训练 (1-${NUM_FOLDS}折)..."
      python train.py -c "$CONFIG_FILE"
    else
      echo "✅ 所有权重已存在，跳过训练。"
    fi

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
}

# =========================================================================
# 4. 主循环入口
# =========================================================================
for dataset in "${DATASETS[@]}"; do

    for model in "${STANDARD_VITS[@]}"; do
        EXP_NAME="${dataset}_vit_${model}"
        run_experiment "$EXP_NAME" "$model" "none" "none" "none"
    done

    for decoder in "${USFM_DECODERS[@]}"; do
        for mode in "${USFM_MODES[@]}"; do

            # 🌟 核心修改：这里已经删除了跳过 SegViT + local 的逻辑！

            DECODER_LOWER=$(echo "$decoder" | tr '[:upper:]' '[:lower:]')
            EXP_NAME="${dataset}_usfm_${DECODER_LOWER}_${mode}"

            run_experiment "$EXP_NAME" "USFM" "$decoder" "$mode" "$PRETRAIN_PATH"
        done
    done
done

echo -e "\n\033[1;32m🎉 ALL ViT EXPERIMENTS COMPLETED 🎉\033[0m"························································································````