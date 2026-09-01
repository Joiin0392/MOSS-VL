# Quantizing MOSS-VL

[English](README.md) | [中文](README_zh.md)

This guide explains how to quantize MOSS-VL with the same recipes behind our released FP8 and NF4 checkpoints, and how to apply them to your own fine-tuned version. All recipes are post-training quantization: they convert a finished checkpoint directly and do not require calibration data or retraining.

Pre-quantized checkpoints are available on Hugging Face and ModelScope:

| Checkpoint | Weight scheme | Engines |
|-|-|-|
| MOSS-VL-Instruct-0708-FP8 | FP8-Dynamic W8A8 | Transformers, SGLang |
| MOSS-VL-Instruct-0708-NF4 | NF4 W4 Keep-4 | Transformers |
| MOSS-VL-Realtime-FP8 | FP8-Dynamic W8A8 | Transformers, SGLang |
| MOSS-VL-Realtime-NF4 | NF4 W4 Keep-4 + HQQ KV8 | Transformers |

## 0. Quick start: quantize your own checkpoint

The scripts in this directory reproduce the released checkpoints end to end. Point them at any MOSS-VL-format checkpoint, including your own fine-tuned or SFT version — no calibration data needed:

```bash
# 1) Weight quantization — pick ONE
#    FP8-Dynamic — recommended, runs on both Transformers and SGLang
python quant/quantize_fp8_dynamic.py --source /path/to/your-checkpoint --output ./my-model-fp8
#    or NF4 Keep-4 — lowest memory, Transformers only
python quant/quantize_nf4_keep_ends.py --source /path/to/your-checkpoint --output ./my-model-nf4 --verify-reload

# 2) Package runtime HQQ INT8 KV cache — optional but recommended
python quant/package_kv8_hqq.py --source ./my-model-fp8 --output ./my-model-fp8-kv8
```

Run inference with the quantized checkpoint exactly like any other checkpoint: see [`inference/`](../inference/README.md) for offline image/video generation and [`realtime_inference/`](../realtime_inference/README.md) for streaming.

Both weight scripts read `text_config.num_hidden_layers` and `text_config.cross_attention_layers` from the checkpoint's `config.json` to decide the quantization scope automatically; use `--num-layers` / `--cross-layers` / `--keep-end-layers` to override. See recipes [A](#2-recipe-a-fp8-dynamic) and [B](#3-recipe-b-nf4-keep-4) for the rationale, and [Section 7](#7-environment) for the required packages.

## 1. What you can quantize

Inference memory is not one thing. Before quantizing, separate:

| Component | What it is | Memory behavior | Our choice |
|-|-|-|-|
| Weights (W) | Linear weights of self-attention, MLP, vision tower, `lm_head` | Permanent; defines checkpoint size | W8-FP8 or W4-NF4 on **language-model Linears only** |
| Activations (A) | Inputs received by quantized kernels | Transient per forward pass | FP8 per-token dynamic (FP8 recipe) or kept BF16 (NF4 recipe) |
| KV Cache | History key/values in attention | Grows with sequence length and video duration | HQQ INT8 KV8 for Transformers; engine-native KV for SGLang |
| Other runtime state | Attention scratch, vision features, allocator | Not in checkpoint | `flash_attention_2` — memory optimization, not quantization |

The single most important rule for MOSS-VL: **quantize the language model selectively and keep multimodal-sensitive modules in BF16.** In our ablations, quantizing the vision tower, cross-attention layers, or `lm_head` — or extending W4 to *all* text layers — consistently broke quality: hallucinated facts, OCR failures, instruction-following regressions, long repetitions.

## 2. Recipe A: FP8-Dynamic

One checkpoint that runs on both Transformers and SGLang, and needs **no calibration data** — weight scales come from weight statistics, activations are quantized dynamically per token at runtime. This is the recipe we recommend for an SFT'd checkpoint.

- Scheme: compressed-tensors `FP8_DYNAMIC` — weights `float8_e4m3fn` channel-wise static, input activations FP8 per-token dynamic.
- Targets: the `q_proj/k_proj/v_proj/o_proj` and `gate_proj/up_proj/down_proj` Linears of the language-model layers that have **no cross-attention**. MOSS-VL-0708 has 48 language layers with cross-attention at indices `2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46`; the remaining 36 layers contribute 252 Linears.
- Keep BF16: vision tower and merger, all cross-attention layers, embeddings, norms, `lm_head`.

The conversion is a one-shot [LLM Compressor](https://github.com/vllm-project/llm-compressor) call:

```python
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/path/to/your-sft-checkpoint"
CROSS_LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46)
SELF_LAYERS = tuple(i for i in range(48) if i not in CROSS_LAYERS)
GROUP = "|".join(str(i) for i in SELF_LAYERS)
TARGET_REGEX = (
    r"re:^model\.language_model\.layers\."
    rf"({GROUP})\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
)
quantized = oneshot(
    model=model,
    recipe=QuantizationModifier(targets=[TARGET_REGEX], scheme="FP8_DYNAMIC"),
    output_dir=None,
    trust_remote_code_model=True,
    precision="bfloat16",
    save_compressed=True,
)
quantized.save_pretrained("/path/to/output", save_compressed=True, safe_serialization=True)
AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True).save_pretrained("/path/to/output")
# then copy over the remaining support files (processor configs, modeling code, chat template, ...)
```

For SGLang compatibility, extend the saved `config.json`'s `quantization_config` with the fused-module mapping and SGLang-side target/ignore regexes:

```json
"packed_modules_mapping": {
  "qkv_proj": ["q_proj", "k_proj", "v_proj"],
  "gate_up_proj": ["gate_proj", "up_proj"]
}
```

And add to `ignore`: `re:^model\.visual\.`, the cross-attention layers under `model.language_model.model.layers`, and `model.language_model.lm_head`.

This is the scheme behind the released MOSS-VL-Instruct-0708-FP8 checkpoint.

## 3. Recipe B: NF4 Keep-4

[BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) NF4 weight-only 4-bit with double quantization and BF16 compute — the lowest-memory recipe, Transformers-only.

- Targets: self-attention and MLP Linears of the **middle 40 language layers** (240 Linears).
- Kept BF16: the **first 4 and last 4** language layers, all cross-attention layers, vision tower, embeddings, norms, `lm_head`. Keeping the endpoint layers full-precision noticeably improves stability versus aggressive all-layer W4.

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

keep = 4
skip_modules = ["model.visual", "cross_attn", "lm_head"] + [
    f"model.language_model.layers.{i}"
    for i in list(range(keep)) + list(range(48 - keep, 48))
]
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/your-sft-checkpoint",
    trust_remote_code=True,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=skip_modules,
    ),
    low_cpu_mem_usage=True,
)
model.save_pretrained("/path/to/output", safe_serialization=True)
```

## 4. KV Cache quantization at runtime

Weight quantization alone is often not enough for long videos — the KV cache keeps growing with the stream. KV quantization is a **generation-time** setting, so it applies unchanged to any fine-tuned checkpoint.

**Transformers — HQQ INT8 KV8.** Package the checkpoint with a quantized-cache generation config:

```json
"cache_implementation": "quantized",
"cache_config": {
  "backend": "hqq", "nbits": 8,
  "axis_key": 0, "axis_value": 0,
  "q_group_size": 64, "residual_length": 128
}
```

inside `generation_config.json` — this requires the `hqq` package. One caveat: with Transformers' `QuantizedCache`, older entries live in `_quantized_keys/_quantized_values` while only a short BF16 tail sits in `keys/values`, so MOSS-VL's cross-attention cache read path must be patched to dequantize the history and concatenate it with the residual tail; see `modeling_moss_vl.py` in the released KV8 checkpoints for the exact patch.

**SGLang.** Use the engine's own KV dtype instead of HQQ, e.g. `--kv-cache-dtype fp8_e4m3` with the FP8 checkpoint.

Do **not** use KV4: KV4 caused repetition, truncation, or runtime failures in our tests.

## 5. Realtime model specifics

For sustained 1 fps streaming, combine FP8 or NF4 Keep-4 weights with HQQ KV8 and `flash_attention_2`. Weight quantization or an attention-backend switch alone is not enough — over long streams the KV cache and attention scratch dominate.

## 6. What did not work

- **AWQ / GPTQ / RTN W4** weight-only on language layers: image/video quality and instruction following degraded; none passed acceptance even with matched calibration budgets.
- **Expanding coverage to all 336 text Linears** or adding `lm_head` / the vision tower to the quantized set: long repetitions and obvious item misses.
- **INT8 dynamic W8A8**: no memory win over baseline in our setup.
- **KV4**: quality/compatibility failures, as above.

## 7. Environment

The quantization stack is the repository's `requirements.txt` plus [`quant/requirements.txt`](requirements.txt), pinned to the versions that produce the released checkpoints:

```bash
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
pip install -i https://pypi.org/simple -r quant/requirements.txt
# FP8 conversion only; see quant/requirements.txt for why --no-deps is needed
pip install -i https://pypi.org/simple --no-deps llmcompressor==0.10.0
```
