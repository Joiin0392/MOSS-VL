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

If `--output` is omitted, the script writes results to `<input_stem>_results.json`.

## Input Format

The script accepts either:

- `messages`
- `prompt` with optional `images` and `videos`

The provided examples use `messages`. Example:

```json
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
```

Relative media paths are resolved relative to the JSON file location.

## Files

- `image_queries.json`: image and multi-image examples
- `video_queries.json`: video example
- `batch_queries.json`: text-only example
- `assets/images`: demo images
- `assets/videos`: demo videos
