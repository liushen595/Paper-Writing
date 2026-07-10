# 研究方法论 (Methodology)

> 本文档记录论文《犯罪意图识别框架 ThreatWeaver》的研究方法论，随方法演进更新。当前版本对应 `docs/Plan.md` 完善后的方案。

## 1. 研究问题

### 1.1 核心问题
在**不依赖敏感词词表**的前提下，识别互联网文本中的**隐式犯罪意图**（implicit criminal intent）——即通过委婉语、迂回、反讽、隐喻、反问等手段表达、能绕过现有毒性检测器的潜在犯罪言论。

### 1.2 痛点（Key Issue）
现有检测模型（如 `unitary/toxic-bert`、Perspective API）依赖词法特征，对**隐式、不含敏感词**的有害文本漏报严重。Wen et al. (2023) 实证：GPT-3.5 零样本对 5 个主流分类器的攻击成功率达 58–96%。

### 1.3 研究目标
训练一个**本地端到端小语言模型（SLM, Qwen3-8B）**，使其：
1. 通过**显式思维链（Explicit CoT）**学会对隐式意图的逻辑溯因；
2. 通过**直接偏好优化（DPO）**学会在模棱两可言论上保持克制（降低误报 FPR）；
3. 通过**隐式思维链内化（Implicit CoT via Stepwise Internalization）**将推理过程压缩进隐状态，推理阶段不输出中间 token，大幅降低延迟。

替代高成本且存在隐私泄露风险的云端 API 调用方案，可在单张 RTX 4090 (24GB) 上完成训练与推理。

## 2. 相关工作与理论支撑

| 方向 | 文献 | 支撑点 |
|---|---|---|
| 隐式毒性检测痛点 | Wen et al., EMNLP 2023 | 现有分类器对隐式毒性漏报；提供 SFT→RM→RL 模板、三分类偏好方案、语言学特征提示、集成奖励 |
| 隐式有害文本检测+解释 | Hoang et al., NAACL 2024 (ToXCL) | pooled 分类头 + CLM 联合损失避免误差传播；目标生成前缀；RoBERTa KL 蒸馏；Conditional Decoding Constraint；自定义解释评估 |
| 隐式 CoT 内化（主方法） | Deng et al., NeurIPS 2024 | Stepwise Internalization：线性移除调度 + Removal Smoothing λ=4 + 优化器重置 + 左移除；Mistral-7B GSM8K 0.51 超 GPT-4 无 CoT 的 0.44，11× 加速 |
| 隐式 CoT 内化（前沿，未来工作） | Zelikman et al., COLM 2024 (Quiet-STaR) | 元 token + mixing head + REINFORCE；完整版需 8×H100，单卡不可行，列为后续扩展 |
| LLM-as-a-Judge | Zheng et al., NeurIPS 2023 | GPT-4 judge 与人类一致率 85%；位置交换一致性过滤；参考引导打分；微调开源三分类 judge；S1/S2 指标；偏差监控 |

## 3. 数据

### 3.1 数据来源
- **源数据**：`crawler/output/doj_raw.jsonl`（美国司法部新闻稿全量爬取结果，不再预过滤犯罪/非犯罪）。
- **关键澄清**：DOJ 新闻稿是**已发生的犯罪叙事或非犯罪事务公告**，并非"隐式意图言论"，不能直接作为训练样本。

### 3.2 数据合成（Phase 0，Teacher Distillation）
1. 从 `doj_raw.jsonl` 加载全量 DOJ 记录，抽取案情要素（罪名类型、标题、摘要）。
2. 调用免费 Teacher LLM API（GLM-4-Flash / Gemini-2.0-Flash）。LLM 首先判断新闻稿是否为刑事案件：
   - **刑事案件**：按 Wen et al. (2023) 的**语言学特征提示**（委婉 euphemism / 迂回 circumlocution / 反讽 sarcasm / 隐喻 metaphor / 反问 rhetorical question）改写为：
     - `implicit_threat`：不含敏感词的隐式意图言论（正样本）。
     - `hard_negative`：话题相近、词汇相近但语境安全的对照言论（硬负样本）。
     - `thought_process`：显式 CoT 推理链 `[推理] A -> B -> C -> 结论`。
     - `label` / `probability` / `category`；其中 `label="Threat"`，`probability` 为 Teacher 给出的置信（不进入训练损失）。
   - **明显非刑事案件**（民事、政策、行政、报告等）：直接生成一条 Safe 噪声样本：
     - `implicit_threat`：复用为中性/安全文本（作为 Safe 训练样本的输入文本）。
     - `hard_negative`：空字符串（非犯罪记录不生成硬负样本）。
     - `thought_process`：说明其为非犯罪事务、无犯罪意图。
     - `label="Safe"`，`probability≈0`，`category="NonCriminal"`。
3. 按 80/20 切分 train/test，test 不参与任何训练。
4. **质量保障**：免费 Teacher 弱于 GPT-4，造数产物须经人工抽检 + judge 一致性过滤才进入训练集。

### 3.3 三分类偏好方案（Wen et al. 2023）
所有样本归入 隐式意图 (Implicit-Intent) > 显式意图 (Explicit-Intent) > 无意图 (No-Intent) 三档，档内视为等价，用于 DPO 偏好对的低分歧标注。

### 3.4 盲测集（Zero-Knowledge Test Set）
- **草垛 (Haystack)**：`data/synthesized/hard_negatives.jsonl`（synthesis 展开的安全对照言论）+ `data/haystack/wildchat_nontoxic.jsonl`（`allenai/WildChat-nontoxic` 英文子集 5000 条，由 `scripts/prepare_haystack.py` 采样）。原 `doj_non_criminal` 已弃用。
- **针 (Needles)**：合成 test.jsonl 中的 `implicit_threat`，与训练集零重叠。
- 随机种子混合，记录 source / ground-truth / 参考推理。

## 4. 模型架构

### 4.1 基座
`Qwen/Qwen3-8B` + **QLoRA**（4-bit NF4 量化 + LoRA 适配器，target=q_proj,v_proj，r=64，α=16）。

### 4.2 ToXCL 风格分类头（Phase 1 创新）
在 LLM 最后隐藏层之上加 **mean-pool 分类头**（二分类 Threat/Safe），与 CLM 损失联合训练：
$$\mathcal{L}_{SFT} = \alpha \cdot \mathcal{L}_{cls} + \beta \cdot \mathcal{L}_{clm}$$
- $\mathcal{L}_{cls}$：pooled hidden state 上的交叉熵。
- $\mathcal{L}_{clm}$：仅对 `thought_process + label` 段计算 Causal LM Loss（prompt 段 label=-100）。
- **动机**：避免纯生成式"标签+解释"拼接的误差传播（ToXCL 实证）。
- **辅助**：在 `thought_process` 前生成犯罪类别前缀（生成式而非固定标签，适配开放类别）。
- **可选增强**：训练 `RoBERTa-Large` Teacher，经 KL 散度蒸馏到分类头。
- **Conditional Decoding Constraint（推理时）**：cls=Safe 直接输出 `[None]`；cls=Threat 生成完整 thought+label。

## 5. 训练流程

### 5.1 Phase 1: SFT
- 目标：基础模型 → 特定任务"逻辑溯因器"。
- 损失：见 §4.2。
- 超参：lr=2e-4, epochs=3, batch=4, grad_accum=4, warmup=0.03, max_seq_len=1024。

### 5.2 Phase 2: DPO（降低误报率核心）
- 目标：让模型在模棱两可言论上保持克制，解决"草木皆兵"。
- 偏好对自动生成（LLM-as-a-Judge, Zheng et al. 2023）：
  - **位置交换一致性过滤**：A/B 顺序交换调用两次，仅一致才采纳。
  - **参考引导打分**：对推理密集型样本提供 ground-truth 标签 + 参考推理（失败率 70%→15%）。
  - **三分类偏好方案**：chosen 为更严谨推理，rejected 为更草率判别。
  - **规则检测器奖励整形**：集成关键词检测器预过滤（Wen et al. 2023 的 `R = R_θ − α·P`）。
- DPO β 起始 **0.1**（Wen et al. 2023 PPO 的 KL 系数经验甜点）。
- **Judge 质量保障**：judge-human 一致性校验（抽样 100 对，目标 S2≥80%）；按 Zheng 2023 App F 微调开源 Qwen3-8B 三分类 judge 作廉价补充。

### 5.3 Phase 3: 隐式 CoT 内化（Stepwise Internalization, Deng et al. 2024）
- **主方法**：从 Phase 1 显式 CoT 模型出发，按线性调度逐步移除 thought token 并微调：
  $$s(t) = \left\lfloor \Delta \frac{t}{T} \right\rfloor$$
- **稳定性三件套**（消融验证）：
  1. **Removal Smoothing**：$s(t)^* = s(t) + o,\; P(o) \propto e^{-\lambda o}$，λ=4（98% o=0，2% 多移除）。
  2. **优化器状态重置**：每次新增移除 token 时重置 AdamW 一阶/二阶矩。
  3. **左移除**：从 thought 序列开头移除（右移除显著更差）。
- Δ 起始 8；目标全部移除后模型直接输出 Threat/Safe + 概率，无中间 token。
- **硬件核算**：RTX 4090 24GB QLoRA 峰值 14-18GB 可行；多轮重训工时长，建议 Δ 适中 + 早停。

## 6. 实验设计

### 6.1 Baselines
1. **toxic-bert**（判别式）：`unitary/toxic-bert`，统计隐式漏报。
2. **qwen-zeroshot**（通用生成式）：未微调 Qwen3-8B 零样本。
3. **explicit-cot**（消融）：Phase 1 显式 CoT 模型（未内化）。
4. **sft-no-dpo**（消融）：Phase 1 后未经 Phase 2，验证 DPO 对 FPR 贡献。
5. **dpo-only**（消融）：Phase 2 DPO 后 LoRA 权重 + SFT 分类头，验证 DPO 对 TPR/FPR 的边际贡献。
6. **implicit-cot**（本方法）：Phase 3 内化模型。

> 原 `roberta-large` baseline 已移除：本任务以 LLM 内化为主路线，RoBERTa 蒸馏源保留为 Phase 1 可选增强，不作单独 baseline。

### 6.2 量化指标矩阵
- 硬阈值判定（prob > 0.5 即 Threat）。
- **FPR** = FP/(FP+TN)：安全评论误判数，要求趋近 0。
- **TPR (Recall)** = TP/(TP+FN)：隐式威胁找出率。
- **F1 / Precision / Accuracy**。
- **混淆矩阵**可视化（图 1）。
- **计算效率**：显式 vs 隐式模型的推理延迟（mean/p95 ms）、Tokens/Second。

### 6.3 质性评估（LLM-as-a-Judge）
- **S1/S2 一致性**（Zheng et al. 2023）：S1 把 tie+不一致算 tie；S2 仅非 tie 样本。目标 S2≥80%。
- **ToXCL 自定义解释评估**（Hoang et al. 2024 Alg.1）：双方均 `[None]` 加分，不匹配罚 0，匹配计 token-F1（正式版外接 BLEU/ROUGE/BERTScore）。
- **偏差监控**：位置偏差（biased-first 率）、冗长偏差、自我增强偏差（避免同族模型既当系统又当裁判）。
- **开源微调 judge**：微调 Qwen3-8B 三分类 judge，统计一致性 16%→65%、格式错误→0% 的提升。

## 7. 结果解释框架（待实验后填充）
- 表 1：盲测 TPR/FPR/F1 对比（预期本方法碾压 toxic-bert）。
- 表 2：推理延迟对比（Explicit vs Implicit CoT），证明内化后提速。
- 图 1：混淆矩阵可视化 FPR 降低。
- 质性图：LLM-judge 评分分布 + 偏差监控雷达图。

## 8. 伦理与局限
- **Faithfulness 警示**（Zelikman et al. 2024）：隐式内化后无法保证隐状态推理可解释，forensic 场景需保留显式模型作为对照。
- **Teacher 噪声**（Wen et al. 2023）：免费 LLM 标注含偏差，需人工抽检。
- **n-gram 指标 inadequacy**（Hoang et al. 2024）：一对多解释关系不能仅靠 BLEU/ROUGE，必须配合 LLM-judge。
- **Judge 偏差**（Zheng et al. 2023）：位置/冗长/自我增强偏差需监控与抑制。
- **训练成本**（Deng et al. 2024）：Stepwise Internalization 移除速率过快导致不收敛，需 Δ 调参。
- **数据范围**：DOJ 新闻稿覆盖的犯罪类型有限，泛化到全部网信犯罪需扩充种子。

## 9. 硬件与可复现性
- 硬件：单张 RTX 4090 24GB。
- 环境：conda env `ML`（包安装用 `conda install`，见 AGENTS.md）。
- 随机种子：42（数据切分、训练、混合盲测集均统一）。
- 配置：`configs/default.yaml` 集中管理所有超参。
- 入口：`scripts/run_all.sh {synth|hardneg|pref|sft|dpo|implicit|blind|eval|all}`。
