
<p align="center">
    <img src="assets/logo.png" width="300"/>
</p>

<div align="center">
    <a href="https://github.com/OpenMOSS/MOSS-VL"><img src="https://img.shields.io/badge/Github-Star-yellow?logo=Github&amp"></a>
    <a href="https://huggingface.co/collections/OpenMOSS-Team/moss-vl"><img src="https://img.shields.io/badge/Huggingface-Download-orange?logo=Huggingface&amp"></a>
    <a href="https://huggingface.co/spaces/OpenMOSS-Team/MOSS-VL"><img src="https://img.shields.io/badge/Huggingface-space-orange?logo=Huggingface"" alt="MOSS-VL-space"></a>
    <a href="https://modelscope.cn/collections/openmoss/MOSS-VL"><img src="https://img.shields.io/badge/ModelScope-Download-blue?logo=ModelScope" alt="ModelScope"></a>
    <br>
    <a href="https://OpenMOSS.github.io/MOSS-VL-Demo/#/"><img src="https://img.shields.io/badge/Website-View-blue?logo=Website&amp"></a>
    <a href="#"><img src="https://img.shields.io/badge/Arxiv-Coming%20Soon-red?logo=Arxiv"></a>
    <a href="assets/feishu.jpg"><img src="assets/feishu-badge.svg" alt="Feishu Join"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="license"></a>

</div>

<p align="center">
    <a href="./README.md"><b>English</b></a> | <a href="./README_zh.md"><b>中文</b></a>
</p>

## MOSS-VL

**MOSS-VL** 是 OpenMOSS 生态系统中核心的多模态模型系列，致力于推动视觉理解技术的前沿发展。针对视频理解中固有的复杂挑战，我们的技术路线图在以下三个关键维度上实施了系统性的扩展（Scaling）策略：

- 📈 **数据扩展 (Data Scaling)**：通过构建大规模、高质量的多模态数据集，驱动模型实现强大的泛化能力。
- 🧠 **参数扩展 (Parameter Scaling)**：提升模型参数规模，以精准捕捉复杂的视觉-语言关联。
- ⏳ **上下文扩展 (Context Scaling)**：延伸时间跨度，实现对长视频内容的深度推理。


---

## 📌 目录
- [🔥 新闻](#-新闻)
- [🏗️ 模型架构](#️-模型架构)
- [🧩 绝对时间戳](#-绝对时间戳)
- [🧬 交叉注意力旋转位置编码 (XRoPE)](#-交叉注意力旋转位置编码-xrope)
- [🎬 演示视频](#-演示视频)
- [📊 训练策略](#-训练策略)
- [📊 评测结果](#-评测结果)
- [🚀 快速上手](#-快速上手)
- [📥 模型下载](#-模型下载)
- [🖥️ SGLang](#️-sglang)
- [📑 路线图与待办事项](#-路线图与待办事项)
- [📜 引用](#-引用)

---

## 🔥 新闻
- **2026/04/24**: 🚀 SGLang 官方已正式支持 MOSS-VL，详见 [sgl-project/sglang](https://github.com/sgl-project/sglang)。
- **2026/04/22**: 🚀 推出基于 SGLang 的 MOSS-VL 推理支持，详见 [`./sglang/`](./sglang/)。
- **2026/04/22**: 🤗 HuggingFace 推理代码更新至最新版本，详见 [MOSS-VL-Base-0408](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Base-0408) 与 [MOSS-VL-Instruct-0408](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0408)。
- **2026/04/08**: 🚀 MOSS-VL-Base-0408 与 MOSS-VL-Instruct-0408 正式开源。
- **2026/04/03**: 🏆 完成 MOSS-VL 全阶段训练 (**PT & SFT**)。
- **2025/10/18**: 🔍 启动了 MOSS-VL 项目。
- **2025/09/30**: ✨ [MOSS-Video-Preview](https://github.com/fnlp-vision/MOSS-Video-Preview) 训练完成。

## 🏗️ 模型架构
**MOSS-VL** 采用基于交叉注意力（Cross-attention）的架构，实现了**视觉编码与认知推理的解耦**。该设计极大地降低了延迟，能够对动态视频流做出即时响应。由于**原生支持模态交错**，模型可在统一的流水线中处理复杂的图像与视频序列，无需繁重的预处理流程。
    
<p align="center">
    <img src="assets/structure.png" alt="MOSS-VL Architecture" width="90%"/>
    <br>
    <em>图 1: MOSS-VL 整体架构图。</em>
</p>

---

## 🧩 绝对时间戳

为了确保模型能够精准感知事件的节奏与时长，**MOSS-VL** 在每一采样帧中均注入了**绝对时间戳**，从而将推理过程锚定在**精确的时间参考基准**之上。

### 📥 输入表示

<p align="center">
    <img src="assets/timestamp_input.svg" alt="时戳序列输入示意图" width="90%"/>
    <br>
    <em>图 2: 带有时间戳的视频序列输入示意图。</em>
</p>

每段视频中都交织插入了**精确的时间标记**。每个时间戳均由**专用特殊 Token**（`<|time_start|>` … `<|time_end|>`）封装，从而明确地锚定了每一帧视觉图像的**时间位置**：
```text
<|im_start|><|vision_start|>
<|time_start|>0.0 seconds<|time_end|><|image_pad|>
<|time_start|>1.2 seconds<|time_end|><|image_pad|>
<|time_start|>2.3 seconds<|time_end|><|image_pad|>
...
<|vision_end|>视频显示了一个具有连续动作的动态场景...<|im_end|>
```

**🌟 设计初衷：**
- **可变帧率适配：** 使用显式时间戳使模型能够处理非均匀采样率，而不丢失时序上下文。
- **精准动作定位：** 引入绝对时间维度，实现细粒度的动作定位，使每一项响应都能精准锚定在特定的时间坐标上。
- **运动动力学分析：** 通过引入时间间隔 ($dt$)，模型能够对运动物理规律进行推理，从而实现对速度、加速度和运动轨迹的精确估算。

---

## 🧬 交叉注意力旋转位置编码 (XRoPE)

MOSS-VL 采用了专门针对其基于交叉注意力的视觉-语言架构而定制的**交叉注意力旋转位置编码 (Cross-attention Rotary Position Embedding, XRoPE)**。该机制将文本 Token 和视频 Patch 映射到一个由时间 (t)、高度 (h) 和宽度 (w) 定义的统一 3D 坐标空间中。

<p align="center">
    <img src="assets/3d-rope.png" alt="MOSS-VL mRoPE 架构示意图" width="80%"/>
    <br>
    <em> 图 3：配备交叉注意力 RoPE (XRoPE) 的 MOSS-VL 架构示意图。</em>
</p>

为了优化跨模态对齐，**XRoPE** 被注入到视觉部分的 **Key (K)** 中以增强位置感知能力，同时保持 **Value (V)** 不变，从而确保特征的保真度。与此同时，XRoPE 被应用于文本部分的 **Query (Q)**，使模型能够通过直接的坐标对齐，精准地检索任意时空区域的信息。


**🌟 核心价值**

- **统一模态建模** —— 通过将时间表达为语言和视频共享的维度，**XRoPE** 实现了在单一坐标系下无缝、连贯的视频-文本推理。
- **精准时空锚定** —— 对齐的 ($t, h, w$) 坐标赋予了模型在 3D 视频空间中定位细小物体和瞬时动作的能力——其精度可细化至每一个补丁（patch）和每一个瞬间。
- **动态输入支持** —— 3D 网格原生适配任意宽高比和分辨率，无需进行固定长度的填充（padding）或遵循严格的输入限制。


---

## 🎬 演示视频

<div align="center">
  <video src="https://gist.github.com/user-attachments/assets/66406aaa-f09f-412c-87b1-97753895ef1f
" width="70%" poster="" controls></video>
<video src="https://gist.github.com/user-attachments/assets/d1ccae33-472f-4d92-96c4-fb6253b07189
" width="70%" poster="" controls></video>
  <p align="center">
    更多示例，请访问我们的 <a href="https://OpenMOSS.github.io/MOSS-VL-Demo/#/">交互式 Demo 页面</a> 🚀<br/>
    在线体验界面：<a href="https://huggingface.co/spaces/OpenMOSS-Team/MOSS-VL">HuggingFace Space</a> 🌐
  </p>
</div>

---

## 📊 训练策略
MOSS-VL 采用多阶段训练方法，逐步构建多模态能力。

<p align="center">
    <img src="assets/total_data_distribution.png" alt="MOSS-VL 训练数据总分布" width="80%"/>
    <br>
    <em>图 4：MOSS-VL 训练数据总分布图。</em>
</p>

### 预训练阶段（PT）
MOSS-VL 通过系统性的**四阶段预训练**，从零开始逐步构建多模态能力：

* **第一阶段 — 视觉-语言对齐** —— 建立视觉特征与语言空间之间的初步桥梁。通过在大规模图文对上进行训练，模型学习将视觉概念与其对应的文本描述联系起来，同时培养基础的 OCR 能力，以理解图像中的文字。

* **第二阶段 — 大规模多模态预训练** —— 扩大模型对海量、多样化多模态语料库的接触面，拓宽模型对世界知识和复杂场景的掌握，为通用智能和高分辨率感知奠定坚实基础。此外，这一阶段还引入了短视频片段，以初步培养视频理解能力。

* **第三阶段 — 高质量多模态预训练** —— 通过在大量高质量的感知、理解和推理数据上进行训练，全面提升模型质量。此阶段结合了细粒度的图像感知、复杂的跨多图理解以及高保真视频推理，旨在强化模型捕捉复杂视觉细节的能力，并掌握丰富多模态输入中的时序关系。

* **第四阶段 — 退火与长上下文外推** —— 将模型的视野延伸至长视频理解，同时通过精心设计的退火（Annealing）策略，利用精选的顶级多模态数据进行训练，将模型的最终性能推向巅峰。

| 阶段 | 策略 | 数据组成 |
| :--- | :--- | :--- |
| **1** | **视觉-语言对齐** | <img src="assets/pt-stage1.png" width="400"/> |
| **2** | **大规模多模态预训练** | <img src="assets/pt-stage2.png" width="400"/> |
| **3** | **高质量多模态预训练** | <img src="assets/pt-stage3.png" width="400"/> |
| **4** | **退火训练与长上下文外推** | <img src="assets/pt-stage4.png" width="400"/> |

### 指令微调 (SFT)
在预训练模型的基础上，**MOSS-VL** 通过**有监督微调 (SFT)** 进行了进一步优化，以对齐人类意图，并全面释放其交互与指令遵循能力。

<p align="center">
    <img src="assets/SFT.png" alt="MOSS-VL SFT 数据组成" width="50%"/>
    <br>
    <em>图 5：MOSS-VL SFT 数据组成图。</em>
</p>


### 人类反馈强化学习 (RLHF)

> [!NOTE]
> MOSS-VL 目前正处于 RLHF 训练阶段。敬请期待后续更新。

---

## 📊 评测结果

我们对 MOSS-VL 在四个关键维度上进行了全面评估：多模态感知、多模态推理、文档/OCR、以及视频理解。评估结果表明，MOSS-VL 取得了卓越的性能，尤其在**通用多模态感知**和**复杂视频分析**方面表现出色。

### 综合表现

下表汇总了 0-100 分制的基准测试评分。总体而言，与 Qwen2.5-VL 和 Qwen3-VL 等行业领先的基准模型相比，**MOSS-VL** 始终稳居第一或第二。

<p align="center">
    <img src="assets/MOSS-VL-Benchmark.png" alt="MOSS-VL Benchmark Comparison" width="90%"/>
    <br>
    <em>图 6: MOSS-VL 与 Qwen 系列在各 Benchmark 上的详细对比。</em>
</p>

### 核心亮点

*   **🚀 领先的视频智能**：MOSS-VL 在视频理解维度取得了 **65.8** 的高分，显著超越 Qwen3-VL（+2pts）。在 `VideoMME`、`MLVU`、`EgoSchema` 以及 `VSI-bench`（**领先 Qwen3-VL-8B-Instruct 达 8.3 个百分点**）等基准测试中，它展现了出色的时序一致性和动作识别能力。
*   **👁️ 卓越的多模态感知**：MOSS-VL 展现了出色的通用图文理解能力，在 `BLINK` 和 `MMBench` 等评测中，其细粒度物体识别和空间推理表现突出。
*   **🧠 稳健的视觉推理**：MOSS-VL 展现了扎实的逻辑推断能力，在 `CVBench` 和 `VisuLogic` 等复杂推理任务中，与 Qwen 系列的最新版本保持高度竞争力。
*   **📄 可靠的文档理解**：虽然模型主要针对通用感知和视频能力进行了优化，但 MOSS-VL 在 OCR 和文档分析方面依然取得了 **83.9** 的成绩，确保了在文本提取和结构化信息处理中的可靠性。

### 评测分析 (Benchmark Analysis)

下图直观展示了 MOSS-VL 在 30 多个专业基准测试中平衡且全面的能力画像。如蓝色实心区域所示，MOSS-VL 实现了最广泛的综合覆盖，尤其在**视频理解**与**多模态感知**象限中展现出了卓越的性能优势。

<p align="center">
  <img src="assets/radar.png" width="600px" alt="MOSS-VL Evaluation Radar">
  <br>
  <em>图 7: MOSS-VL 评测分析图。</em>
</p>

---

## 🚀 快速上手

### 环境配置

```bash
conda create -n moss_vl python=3.12 pip -y
conda activate moss_vl
pip install -i https://pypi.org/simple --no-build-isolation -r requirements.txt
```

### 模型推理

更多可直接运行的推理示例和 Demo 素材，请参阅 [`inference/README.md`](inference/README.md)。
推理支持全模态离线查询，包括纯文本、单图、多图、单视频、多视频，以及基于 `messages` 格式的图视频交错输入。

<details>
<summary><strong>使用 <code>offline_generate</code> 做单条推理</strong></summary>

<br>

```python
import queue
import threading
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

checkpoint = "/path/to/dummy-checkpoint"

processor = AutoProcessor.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    frame_extract_num_threads=1,
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

query = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "path/to/example.jpg"},
                {"type": "text", "text": "描述这张图片。"},
            ],
        }
    ],
    "media_kwargs": {},
    "generate_kwargs": {
        "max_new_tokens": 256,
        "do_sample": False,
        "vision_chunked_length": 64,
    },
}

input_queue = queue.Queue()
output_queue = queue.Queue()
worker = threading.Thread(
    target=model.offline_generate,
    args=(processor, input_queue, output_queue),
    kwargs={"vision_chunked_length": 64},
    daemon=True,
)
worker.start()

input_queue.put(query)
text_chunks = []
while True:
    item = output_queue.get()
    if item in {"<|round_start|>"}:
        continue
    if item == "<|round_end|>":
        break
    text_chunks.append(item)

print("".join(text_chunks))

input_queue.put({"stop_offline_generate": True})
worker.join()
```

</details>

如果需要简单的 batch 离线推理，也可以直接使用 `offline_batch_generate`：

<details>
<summary><strong>使用 <code>offline_batch_generate</code> 做 batch 推理</strong></summary>

<br>

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

checkpoint = "/path/to/dummy-checkpoint"

processor = AutoProcessor.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    frame_extract_num_threads=1,
)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

queries = [
    {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "描述样本 A。"}],
            }
        ],
        "media_kwargs": {},
        "generate_kwargs": {"max_new_tokens": 256, "do_sample": False},
    },
    {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "描述样本 B。"}],
            }
        ],
        "media_kwargs": {},
        "generate_kwargs": {"max_new_tokens": 256, "do_sample": False},
    },
]

with torch.no_grad():
    result = model.offline_batch_generate(
        processor,
        queries,
        vision_chunked_length=64,
    )

texts = [item["text"] for item in result["results"]]
print(texts)
```

</details>

### 微调 (Fine-Tuning)

我们提供了一套基于 HuggingFace `transformers.Trainer` 的轻量级 SFT 微调框架，支持全参数训练与 LoRA，且可独立控制视觉编码器、语言模型和 LM Head 是否参与训练。

```bash
# 全参数 SFT（默认冻结视觉编码器）
bash mossvl_finetune/scripts/run_sft.sh

# LoRA SFT
pip install -i https://pypi.org/simple peft
bash mossvl_finetune/scripts/run_sft_lora.sh
```

训练数据采用与推理查询格式兼容的 JSON 结构，只需额外添加 `response` 字段：

```json
[
  {
    "prompt": "描述这张图片。",
    "response": "一幅群山环绕的美丽风景画。",
    "images": ["path/to/image.jpg"],
    "videos": []
  }
]
```

同时支持多轮对话格式，详细文档请参阅 [`mossvl_finetune/README.md`](mossvl_finetune/README.md)。

---

## 📥 模型下载 

| 模型 | 🤗Download Link | 🤖ModelScope Link |
| :--- | :--- | :--- |
| **MOSS-VL-Base-0408** | [HuggingFace](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Base-0408) | [ModelScope](https://modelscope.cn/models/openmoss/MOSS-VL-Base-0408) |
| **MOSS-VL-Instruct-0408** | [HuggingFace](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0408) | [ModelScope](https://modelscope.cn/models/openmoss/MOSS-VL-Instruct-0408) |

## 🖥️ SGLang

如需查看基于 SGLang 的部署与服务化说明，请参考 [`sglang/README_zh.md`](./sglang/README_zh.md)。



---

## 📑 路线图与待办事项

### ✅ 已达成里程碑 (Milestones)
- [x] **核心架构：** 实现了交叉注意力旋转位置编码 (XRoPE)。
- [x] **高性能基础设施：** 集成了 Megatron-LM 与 CUDA Flash Attention 3。
- [x] **模型发布：** 已开源 `MOSS-VL-Base` 与 `MOSS-VL-Instruct` 模型。
- [x] **模型推理：** 已发布支持图像与视频理解的推理代码。

### 🚀 即将到来 (Upcoming)
- [ ] **训练引擎：** MOSS-VL 的完整训练代码。
- [ ] **实时能力：** 专用的实时视频理解模型。
- [ ] **RL 后训练：** MOSS-VL 系列的人类反馈强化学习 (RLHF)。
- [ ] **技术报告：** 详尽的技术报告与实验分析。

---

## 🤝 致谢
我们衷心感谢 **NVIDIA** 提供的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 框架，以及 **Qwen 团队** 提供的强大的 [Qwen](https://github.com/QwenLM/Qwen) 系列语言模型。这些优秀的开源工作为我们的训练基础设施和核心语言模型奠定了坚实基础。同时，我们也由衷感谢 **SGLang 团队** 提供的高性能 [SGLang](https://github.com/sgl-project/sglang) 推理服务框架，为 MOSS-VL 的高效部署提供了重要支持。

## 📜 引用
```bibtex
@misc{moss_vl_2026,
  title         = {{MOSS-VL Technical Report}},
  author        = {OpenMOSS Team},
  year          = {2026},
  howpublished  = {\url{https://github.com/OpenMOSS/MOSS-VL}},
  note          = {GitHub repository}
}
```

## 🌟 Star History

<a href="https://www.star-history.com/?repos=OpenMOSS%2FMOSS-VL&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=OpenMOSS/MOSS-VL&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=OpenMOSS/MOSS-VL&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=OpenMOSS/MOSS-VL&type=date&legend=top-left" />
 </picture>
</a>

<p align="center">
Built with ❤️ by the <b>OpenMOSS Team</b>
</p>
