# MOSS-VL Fine-Tuning

Supervised fine-tuning framework for MOSS-VL, built on HuggingFace `transformers.Trainer`.

## Directory Structure

```
mossvl_finetune/
├── train.py          # Training entry point
├── data.py           # Dataset and data collator
├── arguments.py      # Argument dataclasses
├── scripts/
│   ├── run_sft.sh        # Full-parameter SFT launch script
│   └── run_sft_lora.sh   # LoRA SFT launch script
└── demo/
    └── sft_data.json     # Example training data
```

## Environment

Use the same environment as the model checkpoint:

```bash
conda create -n moss_vl python=3.12 pip -y
conda activate moss_vl
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
```

For LoRA training, additionally install:

```bash
pip install peft
```

## Data Format

Training data is a JSON list. Two formats are supported:

### Format 1: Prompt / Response (compatible with inference queries)

```json
[
  {
    "prompt": "Describe this image.",
    "response": "A beautiful landscape with mountains and a sunset.",
    "images": ["path/to/image.jpg"],
    "videos": [],
    "system_prompt": "You are a helpful assistant."
  }
]
```

**Automatic Media Placement**

Media placeholders (`<|image|>` and `<|video|>`) are automatically **prepended** to the user message according to the following rules:

* **Images:** Each image consumes a single `<|image|>` placeholder.
* **Videos:**
    * **Plain Paths:** One `<|video|>` placeholder per video.
    * **Segmented Videos:** One `<|video|>` placeholder **per segment** when using the dictionary format:
        ```json
        {"video_path": "...", "segments": [...]}
        ```

### Format 2: Conversations (multi-turn, explicit placeholders)

```json
[
  {
    "conversations": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "<|image|>\nDescribe this image."},
      {"role": "assistant", "content": "A beautiful landscape."},
      {"role": "user", "content": "What is the dominant color?"},
      {"role": "assistant", "content": "Green."}
    ],
    "images": ["path/to/image.jpg"],
    "videos": []
  }
]
```

Multimodal Placeholder Rules

When formatting conversations, you must explicitly include <|image|> or <|video|> placeholders within the content:

- Images: Each image requires exactly one <|image|> placeholder.

- Videos: Standard: Each plain video path consumes one <|video|> placeholder.

- Segmented: Each segment within a video dictionary consumes one <|video|> placeholder.

Backward Compatibility: If your existing data uses a single <|video|> placeholder for a top-level video entry (regardless of segments), the loader will automatically expand it to the correct number of placeholders during the pre-tokenization phase.

### Path Resolution

Relative media paths in the JSON are resolved relative to the JSON file's parent directory (or the `--data_dir` argument if provided).

### Video Entries

Video entries can be provided either as a simple file path string or as an object containing specific time segments:

```json
{
  "videos": [
    "path/to/video.mp4",
    {
      "video_path": "path/to/video.mp4", 
      "segments": [[0, 10], [20, 30]]
    }
  ]
}
```

> **Note:** Because the segmented object in the example above defines two distinct time brackets, it expands into two separate video units. Consequently, you must include two corresponding `<|video|>` placeholders when constructing the training text.

## Usage

> [!NOTE]
> Run from the repository root.

### Full-Parameter SFT

```bash
bash mossvl_finetune/scripts/run_sft.sh
```

### LoRA SFT

```bash
bash mossvl_finetune/scripts/run_sft_lora.sh
```

### Single-GPU Quick Test

```bash
python mossvl_finetune/train.py \
  --model_name_or_path /path/to/checkpoint \
  --data_path mossvl_finetune/demo/sft_data.json \
  --output_dir ./checkpoints/test \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --dataloader_num_workers 0 \
  --gradient_checkpointing True \
  --report_to none
```

## Key Arguments

### ModelArguments

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | (required) | Path to the MOSS-VL checkpoint |
| `--tune_vision` | `False` | Train the vision encoder |
| `--tune_language` | `True` | Train the language model layers |
| `--tune_lm_head` | `True` | Train the LM head projection |

### DataArguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | (required) | Path to the training data JSON file |
| `--data_dir` | auto | Base directory for relative media paths |
| `--max_length` | `4096` | Maximum token sequence length |

### TrainingArguments (extends HF TrainingArguments)

| Argument | Default | Description |
|---|---|---|
| `--vision_chunked_length` | `64` | Chunk size for vision encoding (saves VRAM) |
| `--lora_enable` | `False` | Enable LoRA training |
| `--lora_r` | `64` | LoRA rank |
| `--lora_alpha` | `128` | LoRA alpha |
| `--lora_dropout` | `0.0` | LoRA dropout |
| `--lora_target_modules` | `q_proj,k_proj,v_proj,o_proj` | Comma-separated LoRA target modules |

Plus all standard HuggingFace `TrainingArguments` (`--learning_rate`, `--num_train_epochs`, `--deepspeed`, etc.).

## Module Freeze Control

By default the vision encoder is frozen while the language model and LM head are trained:

```
tune_vision=False   →  vision encoder frozen
tune_language=True  →  all decoder layers trained
tune_lm_head=True   →  output projection trained
```

When LoRA is enabled (`--lora_enable True`), all base parameters are frozen and only the LoRA adapters are trained.

## DeepSpeed

Pass a DeepSpeed config via `--deepspeed`:

```bash
torchrun --nproc_per_node=8 mossvl_finetune/train.py \
  ... \
  --deepspeed ds_config_zero2.json
```

## Label Masking

To ensure the model learns effectively, we apply a specific masking strategy to our training tokens:

- Training Targets: Only the Assistant's responses are used as active training labels.

- Masked Content: System prompts, user queries, and all vision-related tokens (e.g., <|image_pad|>) are assigned an ignore_index=-100 to exclude them from loss calculation.

- EOS Learning: The trailing <|im_end|> token at the end of each Assistant turn is explicitly included in the labels, ensuring the model learns when to stop generating.
