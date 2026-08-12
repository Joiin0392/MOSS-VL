#!/bin/bash
# =============================================================================
# MOSS-VL LoRA Fine-Tuning Launch Script
# =============================================================================
set -euo pipefail

# ---- Distributed config --------------------------------------------------
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$(shuf -i 20000-29999 -n 1)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

# ---- Paths ----------------------------------------------------------------
CHECKPOINT="/path/to/checkpoint"
DATA_PATH="finetune/demo/sft_data.json"
OUTPUT_DIR="./checkpoints/moss_vl_lora"

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
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    \
    --max_length 4096 \
    --vision_chunked_length 64 \
    \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 0 \
    \
    --learning_rate 1e-4 \
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
