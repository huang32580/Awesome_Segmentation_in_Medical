#!/bin/bash

# ==========================================
# 统一全自动化 Benchmark 测试脚本 (支持多数据集 + 全面超参记录)
# 包含: USFM (Local/Official x 3种策略), CNN 系列, TransUnet
# ==========================================

set -e

# ================= 核心配置区 =================
# 你可以在这里添加更多的数据集，如 ("busi" "bus_uc")
DATASETS=("busi")
NUM_FOLDS=5
CONFIG_FILE="config.json"
RESULTS_DIR="results"
# ============================================

mkdir -p "$RESULTS_DIR"
SUMMARY_CSV="${RESULTS_DIR}/master_summary.csv"

# 🌟 初始化汇总表头 (新增了 Dataset 列，总共 10 个超参列)
if [ ! -f "$SUMMARY_CSV" ]; then
    echo "Dataset,Experiment_Name,Model,USFM_Mode,Strategy,Freeze_Mode,Batch_Size,Total_Epochs,Optimizer,Scheduler,Base_LR,Min_LR,Weight_Decay,Layer_Decay,Warmup_Epochs,PA_Avg,DSC_Avg,HD95_Avg,IoU_Avg,GFLOPs,Params" > "$SUMMARY_CSV"
fi

# ==========================================
# 辅助脚本：动态修改配置，并向 Shell 返回当前所有超参
# ==========================================
cat << 'EOF' > modify_config_helper.py
import sys, json, re

model_type = sys.argv[1]
exp_name = sys.argv[2]
freeze_mode = sys.argv[3]
pretrain_status = sys.argv[4]  # "yes" or "no"
usfm_mode = sys.argv[5]        # "local", "official", "none"
dataset_name = sys.argv[6]     # 新增：当前运行的数据集名称

# 1. 加载 config.json
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

# 🌟 动态更新 config.json 里的核心信息
config['name'] = exp_name
config['arch']['type'] = model_type
config['trainer']['checkpoint_dir'] = f"checkpoints_{exp_name}/"
config['trainer']['freeze_mode'] = freeze_mode
config['data']['datasets'] = [dataset_name] # 强制使用当前的数据集

# 2. 提取基础超参
bs = config['data'].get('batch_size', 16)
epochs = config['trainer'].get('epochs', 800)
opt_type = config['optimizer'].get('type', 'AdamW')
sch_type = config['lr_scheduler'].get('type', 'CosineAnnealingLR')
base_lr = config['optimizer']['args'].get('lr', 1e-4)
min_lr = config['lr_scheduler']['args'].get('eta_min', 1e-5)
weight_decay = config['optimizer']['args'].get('weight_decay', 1e-5)
layer_decay = 1.0
warmup_epochs = 0

usfm_mode_decay = 0.65

if model_type == "USFM_UPerNet":
    if pretrain_status == "no":
        config['usfm_args']['PRETRAIN_CKPT'] = None
        usfm_mode_decay = 1.0
    else:
        config['usfm_args']['PRETRAIN_CKPT'] = "./pretrained_models/USFM_latest.pth"
        usfm_mode_decay = 0.65

    yaml_path = config['usfm_args']['cfg']
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            text = f.read()

        text = re.sub(r'MODE:\s*[\'"]?(official|local)[\'"]?', f"MODE: '{usfm_mode}'", text)
        text = re.sub(r'LAYER_DECAY:\s*[0-9.]+', f"LAYER_DECAY: {usfm_mode_decay}", text)

        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        print(f"Warning: Failed to modify yaml. Error: {e}", file=sys.stderr)

    if usfm_mode == "official":
        opt_type = "Official_LayerDecay_AdamW"
        sch_type = "Timm_Cosine_Warmup"
        base_lr = 3e-4
        min_lr = 0.0
        weight_decay = 0.05
        layer_decay = usfm_mode_decay
        warmup_epochs = 20
    else:
        layer_decay = 1.0

with open('config.json', 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2)

print(f"{bs},{epochs},{opt_type},{sch_type},{base_lr},{min_lr},{weight_decay},{layer_decay},{warmup_epochs}")
EOF

# ==========================================
# 核心执行函数
# ==========================================
run_experiment() {
    local DATASET=$1
    local MODEL=$2
    local EXP_NAME=$3
    local FREEZE=$4
    local PRETRAIN=$5
    local USFM_MODE=$6
    local STRATEGY=$7

    echo -e "\n\n======================================================="
    echo " 🚀 启动实验: $EXP_NAME"
    echo " 📁 数据集: $DATASET | 🏷️ 模型: $MODEL | 模式: $USFM_MODE"
    echo "======================================================="

    # 动态修改配置文件，加入 DATASET 传参
    local HYPERPARAMS=$(python modify_config_helper.py "$MODEL" "$EXP_NAME" "$FREEZE" "$PRETRAIN" "$USFM_MODE" "$DATASET")

    # 借鉴原脚本：基于 python 再次确认 checkpoint_dir
    local CKPT_DIR=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['trainer']['checkpoint_dir'])")
    CKPT_DIR=${CKPT_DIR%/} # 去除尾部斜杠

    local INDIVIDUAL_CSV="${RESULTS_DIR}/results_${EXP_NAME}_$(basename ${CKPT_DIR}).csv"

    local NUM_CKPTS=0
    if [ -d "$CKPT_DIR" ]; then
        NUM_CKPTS=$(find "$CKPT_DIR" -name "${EXP_NAME}_fold*_best.pth" | wc -l)
    fi

    if [ "$NUM_CKPTS" -lt "$NUM_FOLDS" ]; then
        echo "未发现完整的 $NUM_FOLDS 折权重 (当前进度: $NUM_CKPTS/${NUM_FOLDS})，开始训练..."
        python train.py -c config.json
    else
        echo "发现完整的 $NUM_FOLDS 折权重，跳过训练，直接进入测试评估阶段..."
    fi

    # 测试阶段
    echo "Fold,PA,DSC,HD95,IoU,GFLOPs,Params" > "$INDIVIDUAL_CSV"
    local LAST_GFLOPS="0"
    local LAST_PARAMS="0"

    for fold in $(seq 1 ${NUM_FOLDS}); do
        local CKPT_PATH="${CKPT_DIR}/${EXP_NAME}_fold${fold}_best.pth"

        if [ -f "$CKPT_PATH" ]; then
            echo ">> Testing Fold $fold ..."
            # 借鉴原脚本：使用 || true 防止中断，并使用更稳健的 grep+cut+xargs 提取
            TEST_OUTPUT=$(python test.py -r "$CKPT_PATH" || true)

            PA=$(echo "$TEST_OUTPUT" | grep -w "PA:" | cut -d':' -f2 | xargs || echo "0")
            DSC=$(echo "$TEST_OUTPUT" | grep -w "DSC:" | cut -d':' -f2 | xargs || echo "0")
            HD95=$(echo "$TEST_OUTPUT" | grep -w "HD95:" | cut -d':' -f2 | xargs || echo "0")
            IOU=$(echo "$TEST_OUTPUT" | grep -w "IoU:" | cut -d':' -f2 | xargs || echo "0")
            GFLOPS=$(echo "$TEST_OUTPUT" | grep -w "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
            PARAMS=$(echo "$TEST_OUTPUT" | grep -w "Params:" | cut -d':' -f2 | xargs || echo "0")

            if [ "$GFLOPS" != "0" ]; then LAST_GFLOPS=$GFLOPS; fi
            if [ "$PARAMS" != "0" ]; then LAST_PARAMS=$PARAMS; fi

            echo "${fold},${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> "$INDIVIDUAL_CSV"
        else
            echo "❌ 警告: 找不到权重文件 $CKPT_PATH"
        fi
    done

    # 计算均值
    local AVG_PA=$(awk -F',' 'NR>1 {sum+=$2; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print 0}' "$INDIVIDUAL_CSV")
    local AVG_DSC=$(awk -F',' 'NR>1 {sum+=$3; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print 0}' "$INDIVIDUAL_CSV")
    local AVG_HD95=$(awk -F',' 'NR>1 {sum+=$4; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print 0}' "$INDIVIDUAL_CSV")
    local AVG_IOU=$(awk -F',' 'NR>1 {sum+=$5; cnt++} END {if(cnt>0) printf "%.4f", sum/cnt; else print 0}' "$INDIVIDUAL_CSV")

    echo "✅ [$EXP_NAME] 五折测试完毕! Avg DSC: $AVG_DSC | Avg IoU: $AVG_IOU"

    # 写入大表 (第一列加上了 DATASET)
    echo "${DATASET},${EXP_NAME},${MODEL},${USFM_MODE},${STRATEGY},${FREEZE},${HYPERPARAMS},${AVG_PA},${AVG_DSC},${AVG_HD95},${AVG_IOU},${LAST_GFLOPS},${LAST_PARAMS}" >> "$SUMMARY_CSV"
}

# ==========================================
# 任务队列排期 (增加了对数据集的循环遍历)
# ==========================================

for dataset in "${DATASETS[@]}"; do
    echo -e "\n🔥 开始数据集 [${dataset}] 的全量测试 🔥"

    # ---------------- 阶段 1: USFM_UPerNet 矩阵测试 ----------------
    for mode in "local" "official"; do
        run_experiment "$dataset" "USFM_UPerNet" "${dataset}_usfm_${mode}_fully_tuning" "none" "yes" "${mode}" "fully_tuning"
        run_experiment "$dataset" "USFM_UPerNet" "${dataset}_usfm_${mode}_froze_encoder" "encoder" "yes" "${mode}" "froze_encoder"
        run_experiment "$dataset" "USFM_UPerNet" "${dataset}_usfm_${mode}_from_scratch" "none" "no" "${mode}" "from_scratch"
    done

    # ---------------- 阶段 2: 其余 CNN 和 TransUnet 测试 ----------------
    OTHER_MODELS=("UNet" "AttUNet" "UNetplus" "UNet3plus" "UNeXt" "CMUNet" "CMUNeXt" "TransUnet")
    for model in "${OTHER_MODELS[@]}"; do
        run_experiment "$dataset" "$model" "${dataset}_${model}_standard" "none" "yes" "none" "standard_tuning"
    done

done

# 清理临时文件
rm -f modify_config_helper.py

echo -e "\n🎉 所有实验已执行完毕！请前往 ${RESULTS_DIR}/ 文件夹查看 ${SUMMARY_CSV} 大汇总表！"