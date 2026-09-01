# MOSS-VL 量化指南

[English](README.md) | [中文](README_zh.md)

本文介绍 MOSS-VL 官方量化方案的实现方式——即我们发布的 FP8 和 NF4 量化模型的复现教程——以及如何用自己微调或 SFT 后的 checkpoint 走同样的流程。所有方案均为训练后量化，**不需要校准数据，也不需要重新训练**，直接对已完成的 checkpoint 做转换。

我们已在 Hugging Face 和 ModelScope 发布预量化模型：

| 模型 | 权重量化方案 | 推理引擎 |
|-|-|-|
| MOSS-VL-Instruct-0708-FP8 | FP8-Dynamic W8A8 | Transformers、SGLang |
| MOSS-VL-Instruct-0708-NF4 | NF4 W4 Keep-4 | Transformers |
| MOSS-VL-Realtime-FP8 | FP8-Dynamic W8A8 | Transformers、SGLang |
| MOSS-VL-Realtime-NF4 | NF4 W4 Keep-4 + HQQ KV8 | Transformers |

## 〇、快速开始：量化你自己的 checkpoint

本目录中的脚本可以端到端复现发布的量化模型。把任何一个 MOSS-VL 格式的 checkpoint——包括你自己微调或 SFT 后的版本——直接喂给脚本即可，无需校准数据：

```bash
# 1）权重量化 —— 二选一
#    FP8-Dynamic —— 推荐，一份 checkpoint 同时兼容 Transformers 和 SGLang
python quant/quantize_fp8_dynamic.py --source /path/to/your-checkpoint --output ./my-model-fp8
#    或 NF4 Keep-4 —— 显存最低，仅 Transformers
python quant/quantize_nf4_keep_ends.py --source /path/to/your-checkpoint --output ./my-model-nf4 --verify-reload

# 2）打包运行时 HQQ INT8 KV Cache —— 可选但推荐
python quant/package_kv8_hqq.py --source ./my-model-fp8 --output ./my-model-fp8-kv8
```

量化后的 checkpoint 推理方式与普通 checkpoint 完全一样：离线图片/视频生成见 [`inference/`](../inference/README.md)，流式推理见 [`realtime_inference/`](../realtime_inference/README.md)。

两个权重量化脚本都会从 checkpoint 的 `config.json` 自动读取 `text_config.num_hidden_layers` 与 `text_config.cross_attention_layers` 来决定量化范围，需要用 `--num-layers` / `--cross-layers` / `--keep-end-layers` 覆盖时可直接参数覆盖；范围异常时脚本会直接报错。方案原理见[方案 A](#二方案-afp8-dynamic)与[方案 B](#三方案-bnf4-keep-4)，环境依赖见[第七节](#七环境与依赖)。

## 一、先分清量化对象

推理显存不是单一组成。量化前请先区分：

| 组成 | 模型中的内容 | 显存特征 | 我们的选择 |
|-|-|-|-|
| 权重 W | self-attention、MLP、视觉塔、`lm_head` 等 Linear 权重 | 常驻显存，决定 checkpoint 大小 | **仅对语言模型 Linear** 做 W8-FP8 或 W4-NF4 |
| 激活 A | 被量化算子接收的输入激活 | 每次前向动态产生 | FP8 方案用 FP8 per-token 动态量化；NF4 方案保持 BF16 |
| KV Cache | attention 保存的历史 key/value | 随序列长度和视频时长增长 | Transformers 侧用 HQQ INT8 KV8；SGLang 侧用引擎自身的 KV 类型 |
| 其他运行时状态 | attention 中间张量、视觉特征、显存分配器 | 不在 checkpoint 中 | `flash_attention_2` —— 属于显存优化，不是量化 |

对 MOSS-VL 最重要的一条规则是：**选择性量化语言模型部分，多模态敏感模块保持 BF16**。我们的消融实验证明，量化视觉塔、cross-attention 层或 `lm_head`，或把 W4 覆盖扩大到全部文本层，都会稳定地破坏质量：事实幻觉、OCR 失败、指令跟随退化、长重复。

## 二、方案 A：FP8-Dynamic

一份 checkpoint 同时兼容 Transformers 和 SGLang，且**完全不需要校准数据**——权重 scale 由权重统计生成，激活在运行时按 token 动态量化。对 SFT 后的 checkpoint，这是首选方案。

- 方案说明：compressed-tensors `FP8_DYNAMIC` —— 权重 `float8_e4m3fn` channel-wise static，输入激活 FP8 per-token dynamic。
- 量化范围：语言模型中**没有 cross-attention** 的层的 `q_proj/k_proj/v_proj/o_proj` 与 `gate_proj/up_proj/down_proj` Linear。MOSS-VL-0708 共 48 层语言模型，cross-attention 位于第 `2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46` 层，因此剩余 36 层共 252 个 Linear。
- 保持 BF16：视觉塔与视觉 merger、全部 cross-attention 层、embedding、norm、`lm_head`。

转换只需一次 [LLM Compressor](https://github.com/vllm-project/llm-compressor) one-shot 调用：

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
# 最后把其余支撑文件拷贝过来（processor 配置、modeling 代码、chat template 等）
```

如需 SGLang 兼容，在保存出的 `config.json` 的 `quantization_config` 中补充融合模块映射与 SGLang 侧的目标/忽略正则：

```json
"packed_modules_mapping": {
  "qkv_proj": ["q_proj", "k_proj", "v_proj"],
  "gate_up_proj": ["gate_proj", "up_proj"]
}
```

并在 `ignore` 中追加：`re:^model\.visual\.`、`model.language_model.model.layers` 下的 cross-attention 层、以及 `model.language_model.lm_head`。

这是发布的 MOSS-VL-Instruct-0708-FP8 所用的方案。

## 三、方案 B：NF4 Keep-4

[BitsAndBytes](https://github.com/bitsandbytes-foundation/bitsandbytes) NF4 4-bit weight-only，double quant，BF16 计算 —— 显存最低的方案，仅支持 Transformers。

- 量化范围：中间 40 个语言层的 self-attention 与 MLP Linear（共 240 个）。
- 保持 BF16：**首 4 层、末 4 层**、全部 cross-attention 层、视觉塔、embedding、norm、`lm_head`。相比激进的全层 W4，保留首尾层能明显改善稳定性。

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

## 四、KV Cache 运行时量化

仅量化权重对长视频往往不够——KV Cache 会随视频时长持续增长。KV 量化是生成时的配置，因此对任何微调后的 checkpoint 都同样适用。

**Transformers —— HQQ INT8 KV8。** 在 `generation_config.json` 中加入量化 cache 配置：

```json
"cache_implementation": "quantized",
"cache_config": {
  "backend": "hqq", "nbits": 8,
  "axis_key": 0, "axis_value": 0,
  "q_group_size": 64, "residual_length": 128
}
```

需要安装 `hqq` 包。注意一点：Transformers 的 `QuantizedCache` 把较早的 KV 存在 `_quantized_keys/_quantized_values` 中，`keys/values` 只保留一段 BF16 residual 尾部，因此 MOSS-VL cross-attention 的 cache 读取路径需要打补丁：先反量化历史 KV、再与 residual 尾部拼接。最新 KV8 量化 checkpoint 中的 `modeling_moss_vl.py` 包含了该补丁的完整实现，可直接参考。

**SGLang。** 不使用 HQQ，改用引擎自身的 KV dtype，例如对 FP8 checkpoint 使用 `--kv-cache-dtype fp8_e4m3`。

**不要**使用 KV4：KV4 会导致重复、截断或运行失败。

## 五、Realtime 模型的额外要求

持续 1 fps 流式推理需要三者结合：FP8 或 NF4 Keep-4 权重 + HQQ KV8 + `flash_attention_2`。只量化权重或只切换 attention backend 都不够——长流下 KV Cache 和 attention 中间状态占主导。

## 六、不推荐的方案

- **AWQ / GPTQ / RTN W4** weight-only：图片/视频质量与指令跟随均退化，在同等校准预算下也未通过验收。
- **把覆盖范围扩大到全部 336 个文本 Linear**，或把 `lm_head`、视觉塔纳入量化：长重复与明显漏项。
- **INT8 动态 W8A8**：相比基线没有显存收益。
- **KV4**：如上所述，质量与兼容性均失败。

## 七、环境与依赖

量化环境 = 仓库 `requirements.txt` + [`quant/requirements.txt`](requirements.txt)，版本固定为产出已发布模型的那一组：

```bash
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
pip install -i https://pypi.org/simple -r quant/requirements.txt
# 仅 FP8 转换需要；为什么用 --no-deps 见 quant/requirements.txt 注释
pip install -i https://pypi.org/simple --no-deps llmcompressor==0.10.0
```
