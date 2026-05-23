#!/bin/bash
set -e

# Run only the USFM experiments relevant to the official-mode comparison:
#   1. USFM + SegViT + official
#   2. USFM + UPerHead + official
#   3. USFM + UPerHead + local_aux
# Strategies intentionally exclude from_scratch.

DATASETS=("busi")
STRATEGIES=("fully_tuning" "freeze_encoder")

CONFIG_FILE="config_all.json"
NUM_FOLDS=5
PRETRAIN_PATH="./pretrained_models/USFM_latest.pth"
LOCAL_TARGET_SIZE=224
OFFICIAL_TARGET_SIZE=512

RESULTS_DIR="results/vit_true_official_subset"
CHECKPOINT_ROOT="checkpoints/vit_true_official_subset"
mkdir -p "$RESULTS_DIR"

cat << 'EOF' > modify_vit_4_config.py
import sys
import json

config_file = sys.argv[1]
exp_name = sys.argv[2]
arch_type = sys.argv[3]
decoder_type = sys.argv[4] if sys.argv[4] != "none" else None
usfm_mode = sys.argv[5] if sys.argv[5] != "none" else None
strategy = sys.argv[6]
pretrain_ckpt = sys.argv[7]
checkpoint_root = sys.argv[8]
local_target_size = int(sys.argv[9])
official_target_size = int(sys.argv[10])

with open(config_file, "r", encoding="utf-8") as f:
    config = json.load(f)

config["name"] = exp_name
config["arch"]["type"] = arch_type
config["trainer"]["checkpoint_dir"] = f"{checkpoint_root}/{exp_name}"

if "data" not in config:
    config["data"] = {}
config["data"]["target_size"] = official_target_size if usfm_mode == "official" else local_target_size
config["data"]["use_pad"] = False

if decoder_type == "SegViT":
    loss_args = {
        "num_classes": 2 if usfm_mode == "official" else 1,
        "dec_layers": 3,
        "mask_weight": 20.0,
        "dice_weight": 1.0,
        "cls_weight": 1.0,
    }
    if usfm_mode == "official":
        loss_args["official_targets"] = True
    config["loss"] = {"type": "ATMLoss", "args": loss_args}
else:
    config["loss"] = {"type": "DiceBCELoss", "args": {}}

if "usfm_args" not in config:
    config["usfm_args"] = {}

if arch_type == "USFM":
    config["usfm_args"]["decoder_type"] = decoder_type
    config["usfm_args"]["mode"] = usfm_mode
    config["usfm_args"]["PRETRAIN_CKPT"] = pretrain_ckpt

    if usfm_mode == "official":
        config["usfm_args"]["base_lr"] = 3e-4
        config["usfm_args"]["warmup_lr"] = 5e-5
        config["usfm_args"]["min_lr"] = 0.0
        config["usfm_args"]["weight_decay"] = 0.05
        config["usfm_args"]["warmup_epochs"] = 20
        config["usfm_args"]["drop_path_rate"] = 0.1
        config["usfm_args"]["aux_weight"] = 0.4
    else:
        config["usfm_args"]["drop_path_rate"] = 0.0
        if usfm_mode == "local_aux":
            config["usfm_args"]["aux_weight"] = 0.4
else:
    config["usfm_args"]["mode"] = "none"

if "trainer" not in config:
    config["trainer"] = {}
config["trainer"]["freeze_mode"] = "encoder" if strategy == "freeze_encoder" else "none"
config.pop("freeze_mode", None)

with open(config_file, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
EOF

run_experiment() {
    local EXP_NAME=$1
    local ARCH_TYPE=$2
    local DECODER_TYPE=$3
    local USFM_MODE=$4
    local STRATEGY=$5

    echo
    echo "======================================================="
    echo "STARTING EXPERIMENT: ${EXP_NAME}"
    echo "======================================================="

    python modify_vit_4_config.py \
        "$CONFIG_FILE" \
        "$EXP_NAME" \
        "$ARCH_TYPE" \
        "${DECODER_TYPE:-none}" \
        "${USFM_MODE:-none}" \
        "$STRATEGY" \
        "$PRETRAIN_PATH" \
        "$CHECKPOINT_ROOT" \
        "$LOCAL_TARGET_SIZE" \
        "$OFFICIAL_TARGET_SIZE"

    local CHECKPOINT_BASE_DIR="${CHECKPOINT_ROOT}/${EXP_NAME}"
    local NEED_TRAIN=false
    for fold in $(seq 1 $NUM_FOLDS); do
        local EXPECTED_PTH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"
        if [ ! -f "$EXPECTED_PTH" ]; then
            NEED_TRAIN=true
            break
        fi
    done

    if [ "$NEED_TRAIN" = true ]; then
        echo "Missing one or more fold checkpoints. Starting training."
        python train.py -c "$CONFIG_FILE"
    else
        echo "All ${NUM_FOLDS} fold checkpoints already exist. Skipping training."
    fi

    local RESULTS_CSV="${RESULTS_DIR}/results_${EXP_NAME}.csv"
    echo "Fold,PA,DSC,HD95,IoU,GFLOPs,Params" > "$RESULTS_CSV"

    echo "Extracting test metrics."
    for fold in $(seq 1 $NUM_FOLDS); do
        local CHECKPOINT_PATH="${CHECKPOINT_BASE_DIR}/${EXP_NAME}_fold${fold}_best.pth"

        if [ -f "$CHECKPOINT_PATH" ]; then
            echo "Testing fold ${fold}."
            TEST_OUTPUT=$(python test.py -c "$CONFIG_FILE" -r "$CHECKPOINT_PATH" 2>&1 || true)

            PA=$(echo "$TEST_OUTPUT" | grep "PA:" | cut -d':' -f2 | xargs || echo "0")
            DSC=$(echo "$TEST_OUTPUT" | grep "DSC:" | cut -d':' -f2 | xargs || echo "0")
            HD95=$(echo "$TEST_OUTPUT" | grep "HD95:" | cut -d':' -f2 | xargs || echo "0")
            IOU=$(echo "$TEST_OUTPUT" | grep "IoU:" | cut -d':' -f2 | xargs || echo "0")
            GFLOPS=$(echo "$TEST_OUTPUT" | grep "GFLOPs:" | cut -d':' -f2 | xargs || echo "0")
            PARAMS=$(echo "$TEST_OUTPUT" | grep "Params:" | cut -d':' -f2 | xargs || echo "0")

            echo "${fold},${PA},${DSC},${HD95},${IOU},${GFLOPS},${PARAMS}" >> "$RESULTS_CSV"
            echo "Fold ${fold} - DSC: ${DSC} | IoU: ${IOU}"
        else
            echo "ERROR: missing checkpoint: ${CHECKPOINT_PATH}"
        fi
    done

    echo "Saved results to ${RESULTS_CSV}"
}

for dataset in "${DATASETS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        exp_name="${dataset}_USFM_SegViT_official_${strategy}"
        run_experiment "$exp_name" "USFM" "SegViT" "official" "$strategy"
    done

    for mode in "official" "local_aux"; do
        for strategy in "${STRATEGIES[@]}"; do
            exp_name="${dataset}_USFM_UPerHead_${mode}_${strategy}"
            run_experiment "$exp_name" "USFM" "UPerHead" "$mode" "$strategy"
        done
    done
done

echo
echo "All selected USFM experiments are complete."
