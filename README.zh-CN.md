# Duan

### Healthcare System 1

**快速决策 · 量化置信度 · 边缘部署**

Duan 是一个面向医疗场景的 **Decision-Native** 模型。它不生成大段文本，而是在**单次前向**中直接返回结构化决策、各选项的概率分布、置信度。当置信度不足时，Duan 的设计意图是把病例交给人工专家，或交给更具分析能力的 **System 2**。

---

## 目录

1. [模型基座](#1-模型基座)
2. [部署与运行](#2-部署与运行)
3. [训练方法](#3-训练方法)
4. [所用数据](#4-所用数据)
5. [评测](#5-评测)
6. [能力边界](#6-能力边界)
7. [引用](#7-引用)
8. [许可](#8-许可)

---

## 1. 模型基座

Duan 是 **Laya** 的微调模型。Laya 是一个开源的**非自回归 System 1 决策引擎**，在单次前向中对 100+ 种语言给出类型化决策。

> CONVAI INNOVATIONS. *Laya: fast, non-autoregressive System 1 decision engine*. Apache-2.0.  
> <https://github.com/convaiinnovations/laya>

Laya 提供三个官方 checkpoint，Duan 建立在其中的 **multilingual** 版本上：

| Checkpoint                                             | Encoder                |      参数量 | 序列预算（`max_len` / `head_max_len`） | 语种       |
| ------------------------------------------------------ | ---------------------- | -------: | -------------------------------- | -------- |
| `convaiinnovations/laya`                               | ModernBERT-large       |     421M | 512 / 192                        | 英语       |
| **`convaiinnovations/laya-multilingual`** （**Duan基座）** | `jhu-clsp/mmBERT-base` | **322M** | **1024 / 256**                   | **100+** |
| `convaiinnovations/laya-typed-decisions`               | ModernBERT-large       |     421M | 1024 / 256                       | 类型化决策工作流 |

### 1.1 模型架构

Laya 是一类**非自回归**架构，具有以下特点：

- **严格恰当评分规则（strictly proper scoring rules）。** 输出分布不仅用交叉熵训练，还以恰当  
  评分规则为目标，因此probability参数**可以当概率读**。
- **类型化问题 + 单一 scorer。** 每个任务都表达为一个带类型的问题——`choice` / `score` /  
  `noul`——三种题型**共用同一个 scorer**，唯一的结构差异是一个 3 行的类型嵌入  
  （`type_emb = Embedding(3, d)`）。**不存在固定类别数的分类头，**&#x64;ecision head 对每个样本  
  在其自身的选项集上做 masked softmax。
- **选项集是输入的一部分。** 标签空间在推理时被渲染进序列，因此模型并不受限于一个固定的  
  决策列表，可以灵活适配不同的决策场景。

| 题型       | `criteria` 形状              | 渲染形式                       | gold 键空间         |
| -------- | -------------------------- | -------------------------- | ---------------- |
| `choice` | `{label: 描述}`              | `"<label>: <描述>"`          | 各 label          |
| `score`  | `[level 0 描述, level 1, …]` | `"level k: …"`             | `"0" … "n-1"`    |
| `noul`   | `{false: …, true: …}`      | `"false: no, it does not"` | `false` / `true` |

### 1.2 序列模板与 token 长度

```text
[CLS] <type> question: <指令 + 渲染后的选项> [SEP] [MASK] opt … [SEP] <state> [SEP]
```

token 预算**不是随意可设的参数，**&#x5B83;遵循 checkpoint 自带的 `rl_agent_config.json`  
（本基座为 `1024` / `256`）。在本语料上实测，单条最长约 **366 token**。

### 1.3 输出

```json
{
  "decision": "review",
  "probabilities": { "routine": 0.08, "review": 0.84, "urgent": 0.08 },
  "confidence": 0.91,
  "abstain": false
}
```

三种题型的返回体键名不同。`choice` 返回`choice` + `probabilities`；

`score` 返回 `score`（期望等级）+ `probabilities`，没有 `choice` 键；  
`noul` 只返回 `noul` = P(true)，没有 `probabilities` 键。

### 1.4 下载模型

权重发布在两个平台（内容一致），任选其一：

**Hugging Face** — <https://huggingface.co/wipen/Duan>

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download wipen/Duan
```

或 Python：

```python
from huggingface_hub import snapshot_download
snapshot_download("wipen/Duan")
```

**ModelScope** — <https://www.modelscope.cn/models/wipenhan/Duan>

```bash
pip install -U modelscope
modelscope download --model wipenhan/Duan
```

或 Python：

```python
from modelscope import snapshot_download
snapshot_download("wipenhan/Duan")
```

---

## 2. 部署与运行

### 2.1 环境要求

| 项 | 要求 |
| --- | --- |
| Python | 3.10 及以上 |
| 硬件 | 仅需 CPU，无需 GPU |
| 内存 | 约 2 GB（322 M 参数，权重 643 MB） |
| 网络 | 首次运行需下载权重（约 643 MB），见 [§1.4](#14-下载模型) |

### 2.2 Agent 部署

把下面这一行提示词粘贴到任意能执行命令的编码 Agent（Codex、WorkBuddy、Claude Code、Cursor 等），它就会克隆仓库、安装依赖、启动服务并验证：

```
克隆 https://github.com/wipen/Duan 并进入其根目录，部署 Duan 并启动本地 web demo：模型权重来自 https://huggingface.co/wipen/Duan（若 huggingface.co 不通，改从 https://www.modelscope.cn/models/wipenhan/Duan 下载，并把 DUAN_MODEL 指向下载到的本地目录）；执行 pip install -r requirements.txt，再执行 python serve.py，最后打开 http://127.0.0.1:8000 确认 /health 返回 ok、且 demo 能跑通一条示例。
```

Agent 需能访问 PyPI（装依赖）；首次运行还需能访问 Hugging Face 下载权重。上面 prompt 已给出 ModelScope 备用源；若只是 HF 直连不稳，也可改用镜像：`export HF_ENDPOINT=https://hf-mirror.com`。

完成后浏览器打开 <http://127.0.0.1:8000> 即可试用；手动步骤见 [§2.3](#23-手动部署)。

### 2.3 手动部署

#### 2.3.1 安装

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

该文件只声明 `laya[serve]`，pip 会据此一并安装 `torch`、`transformers`、`safetensors`、`huggingface_hub`、`fastapi`、`uvicorn`。

#### 2.3.2 启动服务

```bash
python serve.py                        # 127.0.0.1:8000
python serve.py --port 8080            # 换端口
python serve.py --model <id 或路径>     # 换 checkpoint（HF id 或本地目录）
```

| 环境变量 | 含义 | 默认值 |
| --- | --- | --- |
| `DUAN_MODEL` | HF 模型 id 或本地目录 | `wipen/Duan` |
| `DUAN_HOST` / `DUAN_PORT` | 绑定地址 / 端口 | `127.0.0.1` / `8000` |
| `HF_ENDPOINT` | Hugging Face 镜像 | — |

首次启动会下载并加载权重，CPU 上约 20–30 秒；日志出现 `Uvicorn running on http://127.0.0.1:8000` 即为就绪。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/` | Web demo（Playground） |
| `GET` | `/health` | 健康检查 |
| `POST` | `/predict` | `{"state": …, "questions": …}` → 决策结果 |
| `GET` | `/docs` | 交互式 API 文档 |

#### 2.3.3 试用 demo

浏览器打开 <http://127.0.0.1:8000>，在 Playground 中：

1. 在左侧 **Example use cases** 里选一条示例（来自测试集的真实用例）；
2. 点 **Run**；
3. 预期看到：决策结果、每个选项的概率条、置信度。右上角可切换 EN / 中文。

也可直接调 API：

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "state": {"age": 67, "chief_complaint": "chest pain", "heart_rate": 118},
  "questions": {"acuity": {"type": "choice",
    "instructions": "What ESI acuity level should this visit be triaged to?",
    "criteria": {"1": "立即抢救", "2": "高风险", "3": "需多项资源",
                 "4": "需一项资源", "5": "无需资源"}}}
}'
```

预期返回：`answers.acuity.choice`（ESI 等级）、`answers.acuity.probabilities`（各等级概率）与 `confidence`。

#### 2.3.4 常见问题

| 现象 | 原因 / 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'laya'` | 未装依赖 → `pip install -r requirements.txt` |
| 首次启动卡在下载 | `huggingface.co` 不通 → `export HF_ENDPOINT=https://hf-mirror.com` |
| `Address already in use` | 端口被占 → `python serve.py --port 8080` |
| 页面能开但 **Run** 报错 | 后端未启动或地址不对 → 先 `python serve.py`；demo 右上角 **Server** 填完整地址，如 `http://127.0.0.1:8000/predict` |
| 单次推理约 0.4 秒 | CPU 正常速度；避免同时发多个请求 |

---

## 3. 训练方法

训练是对多语言基座的**全参数微调**，DDP 分布式，目标函数为 Laya 的 **RLCD**。

### 3.1 目标函数：RLCD

RLCD 以**严格恰当评分规则**为奖励，采用 GRPO 风格策略梯度。具体配置：

| 组件                          |        取值 | 作用                            |
| --------------------------- | --------: | ----------------------------- |
| `group_size`                |         4 | GRPO 基线采样组大小                  |
| `sigma_start` → `sigma_end` | 0.4 → 0.1 | 探索噪声，线性退火                     |
| `w_sph`                     |      0.75 | 球面分数（spherical score）权重       |
| `w_rps`                     |       1.0 | 排序概率分数权重——**仅对 `score` 题型生效** |
| `ce_weight`                 |       1.0 | 软标签交叉熵权重，与策略梯度项相加             |

奖励采用恰当评分规则而非"答对给分"，意味着模型被优化为认识回答的**不确定性**。

### 3.2 优化与批

| 设置                                      | 取值                  |
| --------------------------------------- | ------------------- |
| `epochs`                                | **8**               |
| `micro_batch`                           | 8                   |
| `target_effective_batch`                | **64** 条序列/步        |
| `max_tokens_per_batch`                  | 4096                |
| `lr_encoder`                            | 2.5e-5              |
| `lr_head`                               | 1.0e-4              |
| `weight_decay` / `min_lr` / `grad_clip` | 0.01 / 1.0e-6 / 1.0 |
| 精度                                      | bf16（自动：算力 ≥ 8.0）   |
| 梯度 / head 检查点                           | 均开启                 |
| `freeze_encoder`                        | false——全参数微调        |
| 随机种子 / 标定种子                             | 20260925 / 20260922 |

### 3.3 后训练温度标定

温度缩放发生在训练**之后**，在留出的标定切片上拟合，并且**按桶分别拟合**而非全局一个值。  
桶名由 `(题型, 选项数)` 决定。把混选项数的数据并成一个温度，会把校准误差全部集中在少数那个桶上。

### 3.4 划分与防泄漏

- 语料分为 `train` 与 `test`。
- 数据泄漏剔除： MedQA-Mainland 官方 train 与 test 之间 **500 条完全重复**已从  
  train 中剔除（详见 [§4.4](#44-已修正的数据缺陷)）。

---

## 4. 所用数据

语料为 **8 个公开数据源、中英双语**，统一转换为 Laya 的类型化决策格式。

### 4.1 数据源

| # | 数据源                         | 语种      | 任务              | 选项数  | train / test         | 标签性质       | 许可               |
| - | --------------------------- | ------- | --------------- | ---- | -------------------- | ---------- | ---------------- |
| 1 | NHAMCS ED                   | en      | 急诊分诊（ESI 1–5）   | 5    | 20,000 / 2,000       | 真实临床结局     | MIT              |
| 2 | UCI Diabetes                | en      | 30 天再入院窗口       | 3    | 20,000 / 2,000       | 真实临床结局     | UCI ML Repo      |
| 3 | MedQA (English)             | en      | 医学知识单选          | 5    | 11,450 / 1,145       | **考试标准答案** | 见 §4.3           |
| 4 | MedQA (Simplified Chinese)  | zh-Hans | 医学知识单选          | 5    | 20,000 / 2,000       | **考试标准答案** | 见 §4.3           |
| 5 | MedQA (Traditional Chinese) | zh-Hant | 医学知识单选          | 4    | 12,701 / 1,270       | **考试标准答案** | 见 §4.3           |
| 6 | CARE-Bench                  | en      | 面向患者的分诊与行动建议    | 4    | 790 / 79             | 专家共识分级     | **CC-BY-NC-4.0** |
| 7 | DDXPlus                     | en      | 自动医学诊断与鉴别诊断     | 2–10 | 20,000 / 2,000       | **完全合成**   | CC-BY-4.0        |
| 8 | PubMedQA（PQA-L）             | en      | 结论是否被支持（yes/no） | 2    | 445 / 44             | 文献专家标注     | MIT              |
|   | **合计**                      |         |                 |      | **105,386 / 10,538** |            |                  |

> **第 3–5、7、8 号都不是临床结局数据。** MedQA 的标签是考试标准答案，CARE-Bench 是专家  
> 共识分级，PubMedQA 是文献标注，DDXPlus 的标签来自生成模型。**它们均不支持任何关于临床  
> 预测能力、发生率或因果关系的声明。** 它们进入语料是为了让类型化决策机制在多种语言与  
> 选项数下得到训练。

### 4.2 语料构成

| 项               | 值                                                                                                        |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| train / test 行数 | **105,386 / 10,538**                                                                                     |
| train 语种分布      | en **72,685**（69.0%）· zh-Hans **20,000**（19.0%）· zh-Hant **12,701**（12.0%）                               |
| 配平策略            | 按源等权，每源上限 20,000 行                                                                                       |
| 选项数分布（train）    | 2: 1,435 · 3: 21,597 · 4: 14,920 · 5: **53,007** · 6: 1,586 · 7: 1,288 · 8: 1,556 · 9: 1,208 · 10: 8,789 |
| 语料落盘体积          | train 122.8 MB · test 12.3 MB（jsonl）                                                                     |

### 4.3 许可

模型权重遵循 **CC-BY-NC-4.0，仅限非商业用途**；代码为 **Apache-2.0**（见 [§8](#8-许可)）。

| 数据源            | 许可                   | 对本项目意味着什么    |
| -------------- | -------------------- | ------------ |
| NHAMCS ED      | MIT                  | 可自由使用与再分发    |
| UCI Diabetes   | UCI ML Repository 条款 | 再分发前需核对条款    |
| PubMedQA       | MIT                  | 可自由使用        |
| DDXPlus        | CC-BY-4.0            | 允许商用，**需署名** |
| **CARE-Bench** | **CC-BY-NC-4.0**     | **仅限非商业用途**  |
| MedQA          | 遵循上游数据集条款            | 见其原始分发渠道     |

**本仓库不分发训练语料。** 转换后的语料是 135 MB 的 jsonl，逐行含来源原文。因此语料一律**在本地用转换脚本从原始来源重新生成**，过程确定、种子固定。

### 4.4 已修正的数据缺陷

| 数据源        | 发现的缺陷                                                                                         | 处理方式                                     |
| ---------- | --------------------------------------------------------------------------------------------- | ---------------------------------------- |
| MedQA 中国大陆 | train/test **存在真实跨 split 泄漏，221 条 test 与 train 完全同题**（6.45%）；dev∩test 32 条；train 组内另有 910 条重复 | 从 train 剔除泄漏（共 500 条）及批内重复 1,140 条       |
| MedQA 中国台湾 | 9 条与留出集重叠                                                                                     | 剔除                                       |
| DDXPlus    | 正确病理有 **73.7%** 位于差分列表**第 0 位**                                                               | 选项按内容哈希确定性打乱；首位命中率降至 **17.3%**，与均匀分布期望一致 |
| DDXPlus    | `icd10` 与 `severity` 是**标签的确定性函数**                                                            | 移出输入 state，只留在 meta                      |
| PubMedQA   | 自带基线模型预测（`reasoning_free_pred` / `reasoning_required_pred`）                                   | 以断言保证不进入输入 state                         |
| CARE-Bench | `information_state` 对某一类纯度 100%；`round_role` 最高 84%                                           | 移出输入 state，并打印纯度作为证据                     |
| NHAMCS ED  | 分诊后干预、等候时间、体征缺失模式均为标签代理                                                                       | 移出输入 state                               |

---

## 5. 评测

指标按标签分布选取，而非沿用惯例——单看准确率没有意义：NHAMCS 分诊在本 test 切片上**多数类**就有 **49.8%**。

以下是 **8 epoch 全量微调后**在 `test` 切片（10,538 条）上的实测值。

| 任务 |  n | 准确率 | 多数类基线 | Macro-F1 | ECE |
| -- | -: | --: | ----: | -------: | --: |


| `ed_triage`（ESI 1–5）      | 2,000 | 0.5600 | 0.4975 | **0.3490** | 0.1423 |
| `readmission_window`（3 级） | 2,000 | 0.5810 | 0.5350 |     0.4178 | 0.0391 |
| `ddx_diagnosis`（2–10 选项）  | 2,000 | 0.9905 | 0.1660 |     0.9900 | 0.0071 |
| `medqa_mcq`（5 / 5 / 4 选项） | 4,415 | 0.3647 | 0.2258 |     0.3659 | 0.4564 |
| `care_escalation`（4 级）    |    79 | 0.6456 | 0.3418 |     0.6317 | 0.2951 |
| `pubmedqa`（yes/no）        |    44 | 0.7273 | 0.6364 |     0.6966 | 0.2727 |

---

## 6. 能力边界

1. **不承诺因果推断。** 统计关联类与自发上报类来源无法支持"药物 X 导致事件 Y"，  
   且本项目中此类来源并未用于任何决策声明。
2. **不做发生率估计。**
3. **暂不声明临床性能。** 已实测的数字全部列在 [§5](#5-评测)，并附分母与限制。常被引用的

5 分类 ESI 预测 0.50–0.65 是文献预期区间，**不是本项目的结果**。


4\. **不替代临床判断。** 尤其在高风险少数类上召回天然受限，应当通过`confidence`阈值筛选把这类病例  
转交人工。  
5\. **不对第三方数据重新授权。** 见 [§4.3](#43-许可)；CARE-Bench 为非商业许可，该约束会传递到在其上训练的模型。

---

## 7. 引用

```bibtex
@misc{duan2026,
  title        = {Duan: A Healthcare System 1 Decision Model},
  author       = {Han, Weipeng},
  year         = {2026},
  howpublished = {\url{https://github.com/wipen/Duan}},
  note         = {Work in progress}
}
```

若使用了底层引擎，请一并引用 Laya：

```bibtex
@misc{laya,
  title        = {Laya: fast, non-autoregressive System 1 decision engine},
  author       = {{ConvAI Innovations}},
  howpublished = {\url{https://github.com/convaiinnovations/laya}},
  note         = {Apache-2.0}
}
```

---

## 8. 许可

代码：**Apache-2.0**（见 `LICENSE`）；权重：**CC-BY-NC-4.0，仅限非商业用途**（见 `LICENSE-MODEL`）。

数据集许可独立于本仓库——见 [§4.3](#43-许可)。

---

## 安全声明

Duan 用于**科研与临床决策支持探索**。它不替代具有相应资质的医疗专业人员，不得用于  
未经验证的自主诊断、处方或其他高风险医疗决策。
