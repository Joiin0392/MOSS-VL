# Inference Demo

This directory contains:

- demo query JSON files for image, video, and batch inference
- a `run_inference.py` script that follows the standard inference style from the project `README.md`

## Files

- `image_queries.json`: single-image examples
- `video_queries.json`: single-video examples
- `batch_queries.json`: batch examples for `offline_batch_generate`
- `run_inference.py`: inference entry script
- `assets/images`: local image assets
- `assets/videos`: local video assets

## Environment

Install dependencies from the project root:

```bash
conda create -n moss_vl python=3.12 pip -y
conda activate moss_vl
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
```

## Usage

Run from the repository root or from this directory.

### 1. Image inference

This mode loops over the samples in `image_queries.json` and calls:

```python
model.offline_image_generate(...)
```

Example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/checkpoint \
  --mode image \
  --input inference/image_queries.json
```

### 2. Video inference

This mode loops over the samples in `video_queries.json` and calls:

```python
model.offline_video_generate(...)
```

Example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/checkpoint \
  --mode video \
  --input inference/video_queries.json
```

### 3. Batch inference

This mode loads the full JSON list and calls:

```python
model.offline_batch_generate(processor, queries)
```

Example:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/checkpoint \
  --mode batch \
  --input inference/batch_queries.json
```

## Optional output path

If `--output` is not provided, the script writes to:

```text
<input_stem>_results.json
```

You can also set it explicitly:

```bash
python inference/run_inference.py \
  --checkpoint /path/to/checkpoint \
  --mode image \
  --input inference/image_queries.json \
  --output inference/image_results.json
```

## Input JSON format

Each sample uses the query format extracted from the model's inference API:

```json
{
  "prompt": "Describe this image.",
  "images": ["assets/images/example.png"],
  "videos": [],
  "media_kwargs": {},
  "generate_kwargs": {
    "max_new_tokens": 256,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "do_sample": false,
    "vision_chunked_length": 64
  }
}
```

## Mode constraints

### `image` mode

- each query must contain exactly one image
- each query must contain zero videos

### `video` mode

- each query must contain exactly one video
- each query must contain zero images

### `batch` mode

- the whole JSON file is passed to `offline_batch_generate`
- all queries must share the same `media_kwargs`
- all queries must share the same `generate_kwargs`

This shared-config rule comes from the batch implementation in the model code.

## Path handling

Relative media paths inside the JSON files are resolved relative to the JSON file location.

For example, in `image_queries.json`:

```json
{
  "images": ["assets/images/bill.png"]
}
```

is resolved relative to `inference/image_queries.json`.

## Output format

### Image / Video mode

The result file contains:

- `mode`
- `checkpoint`
- `input`
- `results`

Each item in `results` includes:

- `index`
- `prompt`
- `images`
- `videos`
- `media_kwargs`
- `generate_kwargs`
- `text`
- `elapsed_seconds`

### Batch mode

The result file contains:

- `mode`
- `checkpoint`
- `input`
- `results`
- `session_states`
- `elapsed_seconds`

Each item in `results` includes:

- `index`
- `prompt`
- `images`
- `videos`
- `media_kwargs`
- `generate_kwargs`
- `text`
- `input_text`
