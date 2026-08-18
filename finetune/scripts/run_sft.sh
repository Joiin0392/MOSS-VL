#!/bin/bash
# =============================================================================
# MOSS-VL Supervised Fine-Tuning Launch Script
# =============================================================================
set -euo pipefail

# ---- Distributed config --------------------------------------------------
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"
if command -v npu-smi &>/dev/null; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(npu-smi info -l | grep -c 'NPU ID')}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
fi

# ---- Paths ----------------------------------------------------------------
CHECKPOINT="/path/to/checkpoint"
DATA_PATH="finetune/demo/sft_data.json"
OUTPUT_DIR="./checkpoints/moss_vl_sft"

# ---- Launch ---------------------------------------------------------------
torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    finetune/train.py \
    \
    --model_name_or_path "$CHECKPOINT" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    \
    --tune_vision False \
    --tune_language True \
    --tune_lm_head True \
    \
    --max_length 4096 \
    --vision_chunked_length 64 \
    \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 0 \
    \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    \
    --logging_steps 1 \
    --save_steps 500 \
    --save_total_limit 3 \
    \
    --gradient_checkpointing True \
    --report_to none
