# Inference

This directory contains ready-to-run offline inference examples for MOSS-VL.
The script supports full-modality offline inference through `model.offline_generate(...)`, including:

- pure text
- single image
- multiple images
- single video
- multiple videos
- interleaved image-video inputs in the `messages` format

## Run

Image examples:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input inference/image_queries.json
```

Video example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input inference/video_queries.json
```

Text-only example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input inference/batch_queries.json
```

SFT / validation-set example in training `jsonl` format:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/dummy-checkpoint \
  --mode offline \
  --input /path/to/valid.jsonl
```

If `--output` is omitted, the script writes results to `<input_stem>_results.json`.

## Input Format

The input file can be either:

- a JSON list, where each item is one query
- a JSONL file, where each line is one sample

Each query can use either of the following formats:

- `messages`
- `prompt` with optional `images` and `videos`

Optional fields such as `media_kwargs`, `generate_kwargs`, and `system_prompt` are also supported.

For JSONL inputs, the script also accepts the standard SFT training formats documented in `mossvl_finetune/README.md`:

- `messages` or `conversations` with top-level `images` / `videos`
- `prompt` / `response` with optional `images` / `videos`

When a training sample includes assistant targets, the loader automatically trims trailing assistant turns at inference time and keeps the remaining context up to the last user turn. For conversation-style samples that use `<|image|>` or `<|video|>` placeholders in text, the loader also reconstructs structured multimodal `messages` content before calling `offline_generate`.

The provided examples use the `messages` format. Example:

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": [
          { "type": "image", "image": "assets/images/bill.png" },
          { "type": "text", "text": "Describe this image." }
        ]
      }
    ],
    "media_kwargs": {},
    "generate_kwargs": {
      "max_new_tokens": 256,
      "do_sample": false,
      "vision_chunked_length": 64
    }
  }
]
```

The `prompt` format is also supported. Example:

```json
[
  {
    "prompt": "Describe this image.",
    "images": ["assets/images/bill.png"],
    "videos": [],
    "media_kwargs": {},
    "generate_kwargs": {
      "max_new_tokens": 256,
      "do_sample": false,
      "vision_chunked_length": 64
    }
  }
]
```

Relative media paths are resolved relative to the JSON file location.

## Files

- `image_queries.json`: image and multi-image examples
- `video_queries.json`: video example
- `batch_queries.json`: text-only example
- `assets/images`: demo images
- `assets/videos`: demo videos
