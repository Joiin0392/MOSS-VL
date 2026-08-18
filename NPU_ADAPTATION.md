# MOSS-VL NPU 适配记录

本文档记录了将 MOSS-VL 从 NVIDIA CUDA 适配到华为昇腾 NPU (Ascend 910B) 的全部修改。

## 环境信息

| 项目 | 值 |
|------|-----|
| NPU 型号 | Ascend 910B2C |
| NPU 数量 | 16 |
| CANN 版本 | 9.0.0 |
| 架构 | x86_64 |
| CXX ABI | cxx_abi_1 (cxx11abi=True) |
| ATB 路径 | /usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1 |

---

## 一、环境搭建修改

### 1.1 conda channel 修复 ( `/root/.condarc` )

**问题**: 原始 `.condarc` 中配置的阿里云 anaconda 镜像 URL 末尾多加了 `/linux-64`，且阿里云镜像已失效 (404)。

**修改**: 将 channel 从阿里云镜像切换为清华 TUNA 镜像。

```yaml
# 修改前
channels:
  - https://mirrors.aliyun.com/anaconda/cloud/conda-forge/linux-64
  - https://mirrors.aliyun.com/anaconda/pkgs/r/linux-64
  - https://mirrors.aliyun.com/anaconda/pkgs/main/linux-64

# 修改后
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
  - defaults
```

### 1.2 Python 依赖替换

**问题**: 原 `requirements.txt` 依赖 CUDA 版 torch (`torch==2.8.0+cu128`) 和 `flash-attn==2.8.1`，均无法在 NPU 上运行。flash-attn 源码编译还因缺少 nvcc 和 NVIDIA 驱动而失败。

**修改**: 新建 `requirements-npu.txt`，将 CUDA 版依赖替换为 NPU 版：

| 包 | CUDA 版 | NPU 版 |
|----|---------|--------|
| torch | 2.8.0+cu128 | 2.8.0+cpu |
| torchvision | 0.23.0+cu128 | 0.23.0+cpu |
| flash-attn | 2.8.1 | *(移除)* |
| torch-npu | *(无)* | 2.8.0.post5 (新增) |

**安装命令**:
```bash
conda create -n moss_vl python=3.12 pip -y
conda activate moss_vl
pip install -i https://pypi.org/simple --no-build-isolation -r requirements-npu.txt
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**说明**:
- `torch==2.8.0+cpu` (CPU 版) 作为基础，`torch-npu==2.8.0.post5` 在其之上添加 NPU 后端支持。
- `torch_npu` 要求 `torch==2.8.0+cpu` (精确匹配，不能使用 CUDA 版)。
- `flash-attn` 是 CUDA 专有包，NPU 上不可用，已移除。代码中使用 `npu_utils.py` 提供的 stub 和 fallback 机制替代。
- `--extra-index-url` 从 `download.pytorch.org/whl/cu128` + `pypi.nvidia.com` 改为 `download.pytorch.org/whl/cpu`。

### 1.3 安装步骤 (已执行)

1. `conda create -n moss_vl python=3.12 pip -y` (修复 channel 后成功)
2. 卸载 CUDA 版包: `pip uninstall -y torch torchvision flash-attn triton` + 所有 `nvidia-*` 包
3. 安装 CPU 版 torch: `pip install torch==2.8.0+cpu torchvision==0.23.0+cpu --index-url https://download.pytorch.org/whl/cpu`
4. 安装 torch_npu: `pip install torch-npu==2.8.0.post5 --no-deps`
5. 其余依赖 (transformers, accelerate, torchcodec 等) 保持不变

---

## 二、代码适配

### 2.1 新增文件: `npu_utils.py` (设备抽象层)

**目的**: 提供统一的设备接口，自动检测并支持 NPU/CUDA/CPU 三种后端。

**核心功能**:

| 函数 | 说明 |
|------|------|
| `get_device_type()` | 返回 "npu" / "cuda" / "cpu" |
| `get_device_count()` | 返回可用设备数量 |
| `synchronize()` | 调用 `torch.npu.synchronize()` 或 `torch.cuda.synchronize()` |
| `get_default_attn_impl()` | NPU 返回 "eager"，CUDA 返回 "flash_attention_2" |
| `resolve_attn_impl(requested)` | 解析用户请求的 attention backend，自动 fallback |
| `install_flash_attn_stub()` | 在非 CUDA 环境安装 flash_attn stub 模块 |
| `print_device_info()` | 打印设备信息 |

**flash_attn stub 机制**:

MOSS-VL 模型通过 `trust_remote_code=True` 从 HuggingFace 加载，远程代码可能在模块级别 `import flash_attn`。NPU 上该包不存在会导致 `ImportError`。`npu_utils.py` 在导入时自动检测：如果非 CUDA 环境，则安装一个轻量 stub 模块到 `sys.modules["flash_attn"]`，使 import 成功；若运行时实际调用了 flash_attn 函数，则抛出清晰的错误提示切换 `attn_implementation`。

### 2.2 修改: `finetune/train.py`

| 行号 | 修改内容 |
|------|---------|
| 17-18 | 新增 `import sys` + `sys.path.insert` 确保能导入 `npu_utils` |
| 23 | 新增 `import npu_utils` |
| 40 | `torch.cuda.synchronize()` → `npu_utils.synchronize()` |
| 84 | `train()` 函数开头新增 `npu_utils.print_device_info()` |
| 107 | `attn_implementation="flash_attention_2"` → `npu_utils.get_default_attn_impl()` |

### 2.3 修改: `inference/run_inference.py`

| 行号 | 修改内容 |
|------|---------|
| 4-15 | 新增 `import sys` + `sys.path.insert` + `import npu_utils` |
| 109 | `attn_implementation="flash_attention_2"` → `npu_utils.get_default_attn_impl()` |

### 2.4 修改: `realtime_inference/run_online_inference.py`

| 行号 | 修改内容 |
|------|---------|
| 7-18 | 新增 `import sys` + `sys.path.insert` |
| 197 | `--attention-backend` 默认值从 `"flash_attention_2"` 改为 `"auto"` |
| 364-367 | `load_model()` 中新增 `import npu_utils` + `npu_utils.print_device_info()` |
| 383 | `attn_implementation=args.attention_backend` → `npu_utils.resolve_attn_impl(args.attention_backend)` |

### 2.5 修改: `finetune/scripts/run_sft.sh` 和 `run_sft_lora.sh`

**问题**: 脚本使用 `nvidia-smi --list-gpus | wc -l` 检测 GPU 数量，NPU 环境下 `nvidia-smi` 不存在。

**修改**: 改为先检测 `npu-smi` 是否可用，优先使用 NPU；不可用时 fallback 到 `nvidia-smi`：

```bash
# 修改前
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}"

# 修改后
if command -v npu-smi &>/dev/null; then
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(npu-smi info -l | grep -c 'NPU ID')}"
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}"
fi
```

### 2.6 新增文件: `requirements-npu.txt`

NPU 专用依赖文件，内容见 1.2 节。

### 2.7 修改: `README.md` / `README_zh.md` / `realtime_inference/README.md`

- 环境配置部分新增 NPU (CANN) 安装指引
- 实时推理示例新增 `ASCEND_VISIBLE_DEVICES=0` 的 NPU 命令
- `realtime_inference/README.md` 添加 NPU 注意事项提示

---

## 三、文件变更清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `npu_utils.py` | 新增 | 设备抽象层 + flash_attn stub |
| `requirements-npu.txt` | 新增 | NPU 专用依赖 |
| `finetune/train.py` | 修改 | cuda→npu_utils, flash→eager |
| `inference/run_inference.py` | 修改 | flash→eager |
| `realtime_inference/run_online_inference.py` | 修改 | 默认 attention backend 改为 auto |
| `finetune/scripts/run_sft.sh` | 修改 | nvidia-smi→npu-smi fallback |
| `finetune/scripts/run_sft_lora.sh` | 修改 | nvidia-smi→npu-smi fallback |
| `README.md` | 修改 | 新增 NPU 安装和使用指引 |
| `README_zh.md` | 修改 | 新增 NPU 安装和使用指引 |
| `realtime_inference/README.md` | 修改 | 新增 NPU 注意事项 |
| `/root/.condarc` | 修改 (环境) | conda channel 从阿里云改为清华 TUNA |

---

## 四、运行方式

### 4.1 环境准备

```bash
conda activate moss_vl
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 4.2 实时推理 (NPU)

```bash
ASCEND_VISIBLE_DEVICES=0 python realtime_inference/run_online_inference.py \
  --checkpoint OpenMOSS-Team/MOSS-VL-Realtime \
  --source video \
  --video path/to/example.mp4 \
  --sample-fps 1 \
  --playback-speed 1 \
  --max-frames 256
```

### 4.3 离线推理 (NPU)

```python
import torch
import torch_npu  # 必须在导入 transformers 前导入
from transformers import AutoModelForCausalLM, AutoProcessor

import npu_utils

checkpoint = "OpenMOSS-Team/MOSS-VL-Realtime"
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, trust_remote_code=True,
    device_map="auto", torch_dtype=torch.bfloat16,
    attn_implementation=npu_utils.get_default_attn_impl(),
)
```

### 4.4 微调 (NPU)

```bash
bash finetune/scripts/run_sft.sh    # 全参数 SFT
bash finetune/scripts/run_sft_lora.sh  # LoRA SFT
```

脚本会自动检测 `npu-smi` 并设置 NPU 数量。可通过环境变量指定设备:
```bash
ASCEND_VISIBLE_DEVICES=0,1,2,3 bash finetune/scripts/run_sft.sh
```

---

## 五、注意事项

1. **CANN 环境必须先加载**: 每次运行前需 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
2. **attention 实现**: NPU 上使用 "eager" (标准 attention) 而非 "flash_attention_2"。性能会有差异，后续可探索 torch_npu 的 `npu_fusion_attention` 作为优化。
3. **flash_attn stub**: 如果模型远程代码在运行时实际调用了 flash_attn 函数 (而非仅 import)，会抛出错误。此时需要检查模型代码是否尊重 `attn_implementation` 参数。
4. **device_map="auto"**: 经验证，accelerate 1.12.0 正确识别 NPU 设备，`device_map="auto"` 可正常工作。
5. **sglang 目录**: SGLang 是独立推理框架，未在本次适配范围内。如需在 NPU 上使用 SGLang，需参考 SGLang 自身的 NPU 支持文档。
6. **torch_npu 版本**: 当前使用 `2.8.0.post5`，匹配 CANN 9.0.0 和 torch 2.8.0。如 CANN 版本不同，需选择对应的 torch_npu 版本。
