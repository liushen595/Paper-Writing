# ThreatWeaver — 隐式犯罪意图识别框架

基于 **Qwen3-8B + QLoRA** 的端到端小语言模型，通过 **SFT → DPO → 隐式思维链内化（Stepwise Internalization）** 三阶段训练，识别互联网中不含敏感词的隐式犯罪意图。

## 目录结构

```
Paper-Writing/
├── AGENTS.md                 # AI 代理协作规范（环境约束、编码约定）
├── README.md                 # 本文件
├── .env.example              # 环境变量模板（API 配置）
├── .env                      # 实际环境变量（不入库，用户自行配置）
├── .gitignore
│
├── crawler/                  # DOJ 新闻稿爬虫（独立 Python venv，与 ML conda 环境分离）
│   ├── config.py             # 爬虫配置（代理、目标 URL）
│   ├── doj_spider.py         # 爬虫主逻辑
│   ├── run.py                # 爬虫入口
│   └── output/               # 爬取结果
│       └── doj_raw.jsonl     # 全量新闻稿（不再预过滤犯罪/非犯罪）
│
├── docs/                     # 项目文档
│   ├── Plan.md               # 研究计划（含文献溯源、技术方案、评估设计）
│   ├── Methodology.md        # 论文研究方法论记录
│   ├── DevelopLog.md         # 开发日志
│   └── Reference-Paper/      # 参考文献（5 篇）
│
├── src/                      # 核心代码
│   ├── utils/                # 基础工具
│   │   ├── env.py            # .env 环境变量加载（多 Provider 支持）
│   │   ├── config.py         # YAML 配置管理（dataclass + YAML 序列化）
│   │   ├── seed.py           # 全局随机种子
│   │   └── logging.py        # 统一日志（控制台 + 文件）
│   │
│   ├── data/                 # 数据处理流水线
│   │   ├── llm_client.py     # Teacher LLM 客户端（OpenAI 兼容，支持 GLM/Agnes 等）
│   │   ├── doj_loader.py     # DOJ 新闻稿加载与案情要素抽取
│   │   ├── synthesis.py      # Phase 0: 造数（Teacher LLM 改写为隐式意图 + CoT）
│   │   ├── hard_negatives.py # 硬负样本构造（LLM 增强安全言论）
│   │   ├── preference.py     # Phase 2: DPO 偏好对生成（LLM-as-Judge + 位置交换一致性）
│   │   ├── dataset.py        # 数据集类（TrainExample、prompt 模板、tokenization）
│   │   └── blind_set.py      # 盲测集组装（haystack + needles → test_blind.csv）
│   │
│   ├── models/               # 模型定义
│   │   ├── student.py        # Qwen3-8B + QLoRA + ToXCL 分类头
│   │   ├── classifier_head.py # RoBERTa Teacher 分类头（蒸馏源）
│   │   └── judge.py          # 开源微调 Judge（三分类 A/B/tie + S1/S2 一致性）
│   │
│   ├── training/             # 训练循环
│   │   ├── sft_dataset.py    # SFT 数据集（chat template + prompt/assistant 分段 loss）
│   │   ├── sft.py            # Phase 1: SFT（联合损失 cls+clm，支持断点续训）
│   │   ├── dpo.py            # Phase 2: DPO（trl.DPOTrainer，支持断点续训）
│   │   └── implicit_cot.py   # Phase 3: Stepwise Internalization（Removal Smoothing + 优化器重置 + 左移除）
│   │
│   └── eval/                 # 评估
│       ├── metrics.py        # 量化指标（FPR/TPR/F1/混淆矩阵/延迟）
│       ├── baselines.py      # 6 个评估基线（toxic-bert / qwen-zeroshot / explicit-cot / sft-no-dpo / dpo-only / implicit-cot）
│       ├── llm_judge.py      # LLM-as-Judge 质量评估（ToXCL Alg.1 + S1/S2 + 偏差监控）
│       └── run_eval.py       # 评估主入口
│
├── configs/                  # 实验配置
│   └── default.yaml          # 默认配置（含 RTX 4060 / 服务器两套参数注释）
│
├── scripts/                  # 运行脚本
│   ├── run_synthesis.py      # Phase 0: 造数入口
│   ├── run_hard_negatives.py # 硬负样本组装入口
│   ├── run_preference.py     # DPO 偏好对生成入口
│   ├── run_sft.py            # Phase 1: SFT 训练入口
│   ├── run_dpo.py            # Phase 2: DPO 训练入口
│   ├── run_implicit_cot.py   # Phase 3: 隐式 CoT 训练入口
│   ├── run_blind_set.py      # 盲测集组装入口
│   ├── run_eval.py           # 评估入口
│   ├── run_judge_eval.py     # LLM-as-Judge 评估入口
│   └── run_all.sh            # 一键全流程（bash scripts/run_all.sh all）
│
├── tests/                    # 单元测试
│   └── test_logic.py         # 纯逻辑测试（12 个，不依赖 GPU/网络）
│
└── data/                     # 数据目录（生成数据不入库）
    ├── raw/                  # 原始数据引用（crawler/output）
    ├── synthesized/          # Teacher LLM 合成数据（train.jsonl / test.jsonl / hard_negatives.jsonl）
    ├── preference/           # DPO 偏好对（dpo_pairs.jsonl）
    ├── blind/                # 盲测集（test_blind.csv）
    └── cache/                # 缓存
```

## 环境配置

### 1. Conda 环境

```bash
conda create -n ML python=3.10 -y
conda activate ML
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge transformers peft trl bitsandbytes accelerate datasets
conda install -c conda-forge python-dotenv pyyaml requests pytest
```

### 2. API 配置

```bash
cp .env.example .env
# 编辑 .env，填入：
#   AGNES_API_KEY=sk-xxx
#   AGNES_BASE_URL=https://apihub.agnes-ai.com/v1
#   AGNES_MODEL_NAME=agnes-2.0-flash
```

### 3. 硬件要求

| 场景 | GPU | 说明 |
|---|---|---|
| 开发/测试 | RTX 4060 8GB | 仅造数、pytest、小规模验证；训练可能溢出 |
| 完整训练 | RTX 4090 24GB / A6000 48GB | QLoRA 8B 峰值 14-18GB 显存 |

## 训练流程详解

### Phase 0: 数据合成（Teacher Distillation）

**输入**：`crawler/output/doj_raw.jsonl`（DOJ 新闻稿全量爬取结果，不再预过滤犯罪/非犯罪）

**输出**：`data/synthesized/train.jsonl` + `test.jsonl`（80/20 自动划分）

**流程**：
1. 加载每条 DOJ 记录，抽取案情要素（title + summary + body + crime_types）。
2. Teacher LLM（Agnes 2.0 Flash）先判断是否为刑事案件：
   - **刑事案件**：按语言学特征提示（委婉/迂回/反讽/隐喻/反问）改写为隐式意图言论 + 硬负样本。
   - **非刑事案件**：生成 Safe 噪声样本（`label="Safe"`，`probability≈0`，无 `hard_negative`）。
3. 输出字段：`implicit_threat` / `hard_negative` / `thought_process` / `label` / `probability` / `category`。
4. 按 80/20 随机切分为 train/test，test 不参与任何训练。

```bash
python -m scripts.run_synthesis --provider agnes --overwrite
# 可选：--limit 50（调试时只处理前 50 条）
```

**预计耗时**：~1231 条 × 10s/条 ≈ 3.5 小时

### Phase 1: 监督微调（SFT）

**输入**：`data/synthesized/train.jsonl`

**输出**：`checkpoints/sft/`（LoRA adapter + 分类头权重）

**架构创新（ToXCL 风格）**：
- 基座：`Qwen/Qwen3-8B` + QLoRA（4-bit NF4 + LoRA r=64）
- 在最后隐藏层之上加 **mean-pool 分类头**（二分类 Threat/Safe）
- 联合损失：`α·L_cls + β·L_clm`
  - `L_cls`：pooled hidden state 上的交叉熵
  - `L_clm`：仅对 `thought_process + label` 段计算 Causal LM Loss
- 辅助：犯罪类别前缀生成（`[Category: Cyber]`）
- Conditional Decoding：cls=Safe 时直接输出 `[None]`，cls=Threat 时生成完整推理

```bash
python -m scripts.run_sft
```

**断点续训**：中断后重新运行，自动从最新 `checkpoint-XXXX` 继续。`num_epochs` 为总轮数。

### Phase 2: 直接偏好优化（DPO）

**输入**：SFT 模型 + `data/synthesized/train.jsonl`（生成候选回复）

**输出**：`checkpoints/dpo/`（DPO 微调后的 LoRA adapter）

**流程**：
1. 用 SFT 模型对每个 prompt 生成两个候选回复（不同温度采样）。
2. 用 Teacher LLM 作为 Judge 打分，采用偏差抑制策略：
   - **位置交换一致性过滤**：A/B 顺序交换调用两次，仅一致才采纳
   - **参考引导打分**：提供 ground-truth 标签 + 参考推理
   - **三分类偏好方案**：隐式 > 显式 > 无意图
3. 使用 `trl.DPOTrainer` 训练，DPO β=0.1

```bash
# 先生成偏好对
python -m scripts.run_preference --judge agnes
# 再训练
python -m scripts.run_dpo
```

**预计耗时**：偏好对生成 ~1 小时；DPO 训练 ~1-2 小时

### Phase 3: 隐式思维链内化（Stepwise Internalization）

**输入**：SFT 模型（显式 CoT）

**输出**：`checkpoints/implicit_cot/`（内化后的模型）

**核心算法（Deng et al. 2024）**：
从显式 CoT 模型出发，按线性调度逐步移除 `thought_process` 中的 token 并微调：
```
s(t) = floor(Δ · t / T)
```
- Δ=8（每 epoch 移除 8 个 thought token）
- **Removal Smoothing**：`s(t)* = s(t) + o, P(o) ∝ exp(-λo), λ=4`（98% 不变，2% 多移除）
- **优化器重置**：每次新增移除 token 时重置 AdamW 状态
- **左移除**：从 thought 开头移除（右移除效果显著更差）

全部 thought token 移除后，模型输入文本直接输出 Threat/Safe + 概率，不输出中间推理，推理延迟大幅降低。

```bash
python -m scripts.run_implicit_cot
```

**预计耗时**：~8-16 小时（20 epoch × 每 epoch 重训）

### 评估

```bash
# 在盲测集上跑全部对比系统
python -m scripts.run_eval

# 可选：LLM-as-Judge 质量评估
python -m scripts.run_judge_eval --predictions outputs/eval/predictions_explicit-cot.json
```

**对比系统**：
1. `toxic-bert`：域外广义毒性参考 baseline
2. `sft-no-dpo`：未经 DPO 的 SFT 消融
3. `threatweaver`：SFT→DPO 主方法

生成式输出可用 `python -m scripts.diagnose_generation outputs/eval/predictions_sft-no-dpo.json outputs/eval/predictions_threatweaver.json` 做严格末尾标签诊断。该诊断与分类头主指标分开报告。

**量化指标**：FPR / TPR (Recall) / F1 / Precision / Accuracy / 混淆矩阵 / 推理延迟 (ms, Tokens/s)

**质性评估**：LLM-as-Judge S1/S2 一致性、ToXCL 自定义解释评估、偏差监控

### 一键全流程

```bash
bash scripts/run_all.sh all
# 或单步执行：bash scripts/run_all.sh synth|hardneg|pref|sft|dpo|implicit|blind|eval
```

## 参考文献

1. Wen et al., "Unveiling the Implicit Toxicity in Large Language Models", EMNLP 2023
2. Hoang et al., "ToXCL: A Unified Framework for Toxic Speech Detection and Explanation", NAACL 2024
3. Deng et al., "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step", NeurIPS 2024
4. Zelikman et al., "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking", COLM 2024
5. Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023
