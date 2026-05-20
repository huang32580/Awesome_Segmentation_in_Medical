#!/bin/bash
set -e

# =========================================================================
# 1. 实验全局配置
# =========================================================================
DATASETS=("busi")
STANDARD_VITS=("TransUnet")
STRATEGIES=("fully_tuning" "from_scratch" "freeze_encoder")

CONFIG_FILE="config.json"
NUM_FOLDS=5
PRETRAIN_PATH="./pretrained_models/USFM_latest.pth"

# 🚀 结果汇总文件夹
RESULTS_DIR="results/vit_true"
mkdir -p "$RESULTS_DIR"

# =========================================================================
# 2. 辅助 Python 脚本 (配合大一统 Config 完美切换)
# =========================================================================
cat << 'EOF' > modify_vit_config.py
import sys, json

config_file = sys.argv[1]
exp_name = sys.argv[2]
arch_type = sys.argv[3]
decoder_type = sys.argv[4] if sys.argv[4] != "none" else None
usfm_mode = sys.argv[5] if sys.argv[5] != "none" else None
strategy = sys.argv[6]
pretrain_ckpt = sys.argv[7]

with open(config_file, 'r', encoding='utf-8') as f:
    config = json.load(f)

config['name'] = exp_name
config['arch']['type'] = arch_type
config['trainer']['checkpoint_dir'] = f"checkpoints/vit_true/{exp_name}"

# 1. 动态切换损失函数 (SegViT 强绑定 ATMLoss，其余全用 DiceBCE)
if decoder_type == 'SegViT':
    config['loss'] = {
        "type": "ATMLoss",
        "args": {"num_classes": 1, "dec_layers": 3, "mask_weight": 20.0, "dice_weight": 1.0, "cls_weight": 1.0}
    }
else:
    config['loss'] = { "type": "DiceBCELoss", "args": {} }

# 2. 动态切换 USFM 参数及官方专属超参数
if 'usfm_args' not in config: config['usfm_args'] = {}

if arch_type == 'USFM':
    config['usfm_args']['decoder_type'] = decoder_type
    config['usfm_args']['mode'] = usfm_mode
    config['usfm_args']['PRETRAIN_CKPT'] = 'none' if strategy == 'from_scratch' else pretrain_ckpt

    # 🚀 核心修改：如果是 official 模式，自动注入官方专属数值！
    if usfm_mode == 'official':
        config['usfm_args']['base_lr'] = 3e-4        # 官方专属大学习率
        config['usfm_args']['drop_path_rate'] = 0.1  # 官方开启防过拟合
    else:
        config['usfm_args']['drop_path_rate'] = 0.0  # 局部模式关闭
        # 注：local 模式的 lr 由顶层 optimizer 决定，无需在此注入

elif 'usfm_args' in config:
    # 如果是非 USFM 模型 (如 TransUnet)，彻底关闭模式
    config['usfm_args']['mode'] = 'none'

config['freeze_mode'] = 'encoder' if strategy == 'freeze_encoder' else 'none'

with open(config_file, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
EOF

# =========================================================================
# 3. 核心实验执行函数 (已修复自动跳过与文件名)
# =========================================================================
run_experiment() {
    local EXP_NAME=$1
    local ARCH_TYPE=$2
    local DECODER_TYPE=$3
    local USFM_MODE=$4
    local STRATEGY=$5

    echo -e "\n\033[1;32m=======================================================\033[0m"
    echo -e "\033[1;32m 🚀 STARTING EXPERIMENT: [${EXP_NAME}]\033[0m"
    echo -e "\033[1;32m=======================================================\033[0m"

    # 1. 修改配置
    python modify_vit_config.py "$CONFIG_FILE" "$EXP_NAME" "$ARCH_TYPE" "${DECODER_TYPE:-none}" "${USFM_MODE:-none}" "$STRATEGY" "$PRETRAIN_PATH"

    # 🚀 2. 修复后的检查逻辑：匹配 train.py 实际生成的文件路径
    CHECKPOINT_BASE_DIR="checkpoints/vit_true/${EXP_NAME}"
    NEED_TRAIN=false
    for fold in $(seq 1 $NUM_FOLDS); do
      # 💡 匹配规则：实验名/实验名_foldX_best.pth
      EXPECTED_PTH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"
      if [ ! -f "$EXPECTED_PTH" ]; then
        NEED_TRAIN=true
        break
      fi
    done

    if [ "$NEED_TRAIN" = true ]; then
      echo "🔥 权重不齐全，开始训练流程 (内部执行五折)..."
      python train.py -c "$CONFIG_FILE"
    else
      echo "✅ 所有 5 折权重已存在 [${CHECKPOINT_BASE_DIR}]，跳过训练，直接提取指标。"
    fi

    # 3. 自动化测试提取逻辑
    RESULTS_CSV="${RESULTS_DIR}/results_${EXP_NAME}.csv"
    echo "Fold,PA,DSC,HD95,IoU,GFLOPs,Params" > "$RESULTS_CSV"

    echo "--- 🧪 正在从现有权重提取测试指标 ---"
    for fold in $(seq 1 $NUM_FOLDS); do
      # 💡 同样修复这里的测试路径
      CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"

      if [ -f "$CHECKPOINT_PATH" ]; then
        echo "▶ Testing Fold ${fold}..."
        TEST_OUTPUT=$(python test.py -c "$CONFIG_FILE" -r "$CHECKPOINT_PATH" 2>&1 || true)

        # 指标提取逻辑
        PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs || echo "0")
        DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs || echo "0")
        HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs || echo "0")
        IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs || echo "0")
        GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
        PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs || echo "0")

        echo "${fold},${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> "$RESULTS_CSV"
        echo "📊 Fold ${fold} - DSC: ${DSC} | IoU: ${IOU}"
      else
        echo "❌ 错误: 找不到权重文件: $CHECKPOINT_PATH"
      fi
    done
    echo "✅ [${EXP_NAME}] 结果已汇总至: $RESULTS_CSV"
}

# =========================================================================
# 4. 主循环入口 (优先级保持不变)
# =========================================================================
for dataset in "${DATASETS[@]}"; do
    # SegViT 系列
    for mode in "official" "local"; do
        for strategy in "${STRATEGIES[@]}"; do
            EXP_NAME="${dataset}_USFM_SegViT_${mode}_${strategy}"
            run_experiment "$EXP_NAME" "USFM" "SegViT" "$mode" "$strategy"
        done
    done

    # UPerHead 系列
    for mode in "official" "local" "local_aux"; do
        for strategy in "${STRATEGIES[@]}"; do
            EXP_NAME="${dataset}_USFM_UPerHead_${mode}_${strategy}"
            run_experiment "$EXP_NAME" "USFM" "UPerHead" "$mode" "$strategy"
        done
    done

    # TransUnet 基准
    for model in "${STANDARD_VITS[@]}"; do
        EXP_NAME="${dataset}_${model}_baseline"
        run_experiment "$EXP_NAME" "$model" "none" "none" "fully_tuning"
    done
done

echo -e "\n\033[1;32m🎉 所有实验与评估已完成！\033[0m"