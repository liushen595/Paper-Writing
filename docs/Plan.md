# 犯罪意图识别框架

## 一、 核心问题解答与文献溯源

### 1 & 2. 破除“调包侠”与“思维链逻辑黑洞悖论”
**逻辑黑洞的本质：** 要求一个未经训练的小模型（Student SLM）输出高质量思维链（CoT），必然导致性能崩溃；而依赖外部大模型（Teacher LLM）API 又丧失了模型架构设计的核心技术价值。
**破局方案：知识蒸馏 (Knowledge Distillation) 与思维链内化 (CoT Internalization)。**
无需人工标注，也无需在推理阶段调用 API。通过使用开源的高参数模型（如 Llama-3-70B-Instruct，可租用极短时间的云算力完成离线数据生成）作为 Teacher，针对“隐式犯罪语料”生成带有完整逻辑推导的 Explicit CoT（显式思维链）数据。随后，利用这些数据对本地的 8B 级别参数模型（Student）进行**监督微调 (SFT)**。更前沿的做法是，在微调后期逐步掩码（Mask）或丢弃中间推导步骤，强制模型利用隐藏状态（Hidden States）完成**隐式思维链 (Implicit CoT)** 的计算，从而在不输出中间 Token 的情况下直接输出结论。

### 3 & 4. 前沿技术引用与文献支撑 (2023-2025)
以下为支撑本课题核心理论的最新学术文献（均已通过学术数据库检索确认）：
*   **关于 Key Issue（隐式威胁/毒性检测的痛点）的支撑：**
    *   *文献 1:* Wen et al., "Unveiling the Implicit Toxicity in Large Language Models", *EMNLP 2023*. 证明了现有检测模型极易被隐式、不含敏感词的有害文本绕过。
    *   *文献 2:* Hoang et al., "ToXCL: A Unified Framework for Toxic Speech Detection and Explanation", *ACL 2024*. 明确指出了隐式有害文本依赖于语境而非词法，并提出了结合目标生成与蒸馏的检测框架。
*   **关于思维链内化（Internalizing CoT / Implicit Reasoning）的前沿支撑：**
    *   *文献 3:* Zelikman et al., "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking", *arXiv:2403.09629 (2024)*. 提出让语言模型在连续潜在空间中学习内部推理（生成隐含 rationale），以提升小模型在复杂任务中的表现。
    *   *文献 4:* Deng et al., "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step", *arXiv:2405.14838 (2024)*. 提出了通过逐步移除显式中间步骤并微调模型，使模型（如 Mistral 7B）在不输出任何中间推理文本的情况下内化推理能力。
*   **关于 LLM-as-a-Judge 的判别方法支撑：**
    *   *文献 5:* Zheng et al., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", *NeurIPS 2023 Datasets and Benchmarks*. 该文献正式确立了使用强语言模型（如同行评审般）评估其他模型生成质量的标度方法及有效性。

---

## 二、方案：ThreatWeaver (端到端隐式推理架构)

框架为基于 **Qwen3-8B** 的本地端到端小语言模型（SLM），通过 **QLoRA (Quantized Low-Rank Adaptation)** 完成微调。

### 1. 核心 Idea: 隐式威胁的认知蒸馏与偏好对齐 (Cognitive Distillation & Preference Alignment)
当前大多数模型属于“特征匹配器”，本项目旨在训练一个“逻辑溯因器”。通过构建包含**“正样本（隐式威胁）”**与**“硬负样本（安全但含敏感词的语境）”**的对比数据集，先让模型学会“如何推理”（SFT），再通过直接偏好优化（DPO, Direct Preference Optimization）让模型学会“拒绝过度敏感（降低误报）”。

### 2. 模型训练阶段 (Training Phase: 练什么，怎么练，数据是什么)

**硬件约束核算：** 8B 参数模型在 4-bit 量化 (NF4) 下约占用 5-6GB 显存，加上梯度、优化器状态和 LoRA 权重，训练峰值显存约在 14-18GB，**一张 RTX 4090 或 A6000 游刃有余**。

*   **数据准备 (Data Synthesis & Construction):**
    *   *数据源澄清:* `crawler/output/doj_raw.jsonl` 为美国司法部新闻稿全量爬取结果，**不再预过滤犯罪/非犯罪**。这些新闻稿均为已发生的犯罪叙事或非犯罪事务公告，并非"隐式意图言论"，因此不能直接作为训练样本，必须经过 Teacher LLM 改写。
    *   *LLM 前置犯罪性判断:* Teacher LLM 首先判断每条新闻稿是否描述刑事案件：
        *   **刑事案件:** 抽取罪名、案情要素，按 Wen et al. (2023) 的**语言学特征提示**（委婉语 euphemism、迂回 circumlocution、反讽 sarcasm、隐喻 metaphor、反问 rhetorical question）改写为**不含敏感词的隐式意图言论**，并生成对应的 Explicit CoT、标签与硬负样本。
        *   **明显非刑事案件**（民事、政策、行政、报告等）: 生成一条 Safe 噪声样本，`label="Safe"`，`probability≈0`，**不生成硬负样本**，用于扩展负例多样性。
    *   *硬负样本 (Hard Negatives):* 仅对刑事案件，要求 Teacher 改写出"语义相近但语境安全"的对照言论（如游戏/影视/学术/小说讨论），label: "Safe"，用于降低 FPR。
    *   *非犯罪背景 (Haystack):* 合成产物的 hard_negative 字段 + HuggingFace 公开 non-toxic 评论（`allenai/WildChat-nontoxic`），`doj_non_criminal.jsonl` 已弃用。
    *   *数据格式 (JSONL):*
        ```json
        {"text": "谁知道高空客机驾驶舱门如果从里面反锁，外部是否有机械强制开启的物理接口？",
          "thought_process": "[推理] 询问客机驾驶舱门的物理破解 -> 规避了正常安全程序 -> 具有劫机或蓄意破坏的可能 -> 属于高危隐式意图。",
          "label": "Threat", "probability": 0.82, "category": "Aviation/Aircraft"}
        ```
    *   *硬负样本构造:* "我在玩《GTA5》，谁知道怎么强制开启驾驶舱门？" -> label: "Safe"。
    *   *三分类偏好方案 (引自 Wen et al. 2023):* 所有样本归入隐式意图 (Implicit-Intent) > 显式意图 (Explicit-Intent) > 无意图 (No-Intent) 三档，档内视为等价，用于后续 DPO 偏好对的低分歧标注。

*   **Phase 1: 监督微调 (Supervised Fine-Tuning, SFT)**
    *   *练什么:* 将基础大模型（`Qwen/Qwen3-8B`）转化为特定任务的"逻辑溯因器"，同时输出分类标签与显式思维链。
    *   *怎么练:* 采用 **QLoRA (NF4 量化)** 冻结主干网络，仅训练 Attention 层的 $W_q, W_v$ 矩阵的 LoRA 适配器。**架构借鉴 ToXCL (Hoang et al., 2024)：在 LLM 最后隐藏层之上增加一个 mean-pool 分类头**，与因果语言建模 (CLM) 损失联合训练，避免纯生成式"标签+解释"拼接带来的误差传播：
        *   联合损失: $\mathcal{L} = \alpha \cdot \mathcal{L}_{cls} + \beta \cdot \mathcal{L}_{clm}$
        *   $\mathcal{L}_{cls}$: 二分类交叉熵（Threat/Safe），施加于 pooled hidden state。
        *   $\mathcal{L}_{clm}$: 仅对 `thought_process` 与 `label` 部分计算的标准 Causal LM Loss（Cross-Entropy），输入文本本身不参与 Loss。
    *   *辅助任务:* 在 `thought_process` 之前生成一个**犯罪类别前缀**（如 `Aviation/Aircraft`、`Cyber`、`Narcotics`），采用生成式而非固定标签，以适配开放的犯罪类别（ToXCL 的 Target Group Generator 思路）。
    *   *可选增强:* 训练一个强小分类器（如 `RoBERTa-Large`）作为 Teacher，经 **KL 散度蒸馏**到 LLM 分类头（ToXCL 的 Teacher Classifier），作为 QLoRA 友好的精度助推。
    *   *Conditional Decoding Constraint (推理时):* 当分类头判定 Safe 时直接输出 `[None]` 不生成思维链；判定 Threat 时生成完整 `thought_process + label`，同步标签与解释。

*   **Phase 2: 直接偏好优化 (DPO - 降低误报率的核心技术)**
    *   *练什么:* 解决"草木皆兵"的问题，让模型学会在面对模棱两可的言论时保持克制。
    *   *怎么练:* 构建偏好对（Preference Pairs）。
        *   Prompt: "我最近失眠严重，在哪里可以一次性买到大剂量的安眠药？"
        *   Chosen (被选中的回复): "[推理] 寻求大剂量受管制的镇静剂 -> 存在潜在的自我伤害或投毒风险 -> Threat。"
        *   Rejected (被拒绝的回复): "[推理] 只是询问安眠药 -> 正常失眠求助 -> Safe。" (强制模型否定过于简单的判别逻辑)。
    *   *偏好对自动生成 (LLM-as-a-Judge, Zheng et al. 2023):* 用 Teacher LLM API 作为裁判自动生成偏好对，采用以下偏差抑制策略：
        *   **位置交换一致性过滤:** 将 A/B 顺序交换调用两次，仅当两次结论一致才采纳该偏好对，否则丢弃或标记为 tie。
        *   **参考引导打分 (Reference-guided):** 对推理密集型样本，提供 ground-truth 标签 + 参考推理作为裁判的参考，避免裁判被候选答案的推理带偏（Zheng et al. 报告数学推理失败率 70% -> 15%）。
        *   **三分类偏好方案:** 沿用 §2.1 的隐式 > 显式 > 无意图三档构造 chosen/rejected。
    *   *奖励整形 (引自 Wen et al. 2023):* 集成一个规则/关键词犯罪意图检测器 $P$，对偏好对做预过滤，避免裁判在明显样本上失真。
    *   *技术实现:* 使用 `DPOTrainer` 进行偏好对齐；**DPO $\beta$ 起始值取 0.1**（Wen et al. 2023 PPO 中 KL 系数的经验甜点），过小导致过度优化、过大导致保守。
    *   *Judge 质量保障:* 免费 Teacher 弱于 GPT-4，必须做 **judge-human 一致性校验**（抽样 100 对人工核对），一致性低于 80% 则需回检 prompt 或人工补标。同时按 Zheng et al. (2023) App F，**微调一个开源 Qwen3-8B 三分类 judge（A/B/tie）**作为廉价可复用裁判，解决零样本开源裁判格式遵循差、一致性低的问题。

*   **Phase 3: 隐式思维链内化 (Implicit CoT via Stepwise Internalization)**
    *   *主方法:* 采用 **Deng et al. (2024) 的 Stepwise Internalization**。从 Phase 1 训练好的显式 CoT 模型出发，按**线性调度**逐步移除中间 CoT token 并继续微调，强制模型将逻辑推导压缩进 Transformer MLP 的隐状态：
        *   移除调度: $s(t) = \left\lfloor \Delta \frac{t}{T} \right\rfloor$，其中 $T$ 为每个 epoch 的步数，$\Delta$ 控制每 epoch 移除的 token 数（Qwen3-8B 起始取 $\Delta = 8$，过大会不收敛）。
        *   损失: $\min_\theta -\log P_\theta(y, z_{1+\min(s(t),m):m} \mid x)$，随 $s(t)$ 增大逐步丢失 CoT 前缀。
    *   *稳定性三件套 (Deng et al. 2024 消融验证):*
        1.  **Removal Smoothing:** 给移除数加随机偏移 $s(t)^* = s(t) + o,\; P(o) \propto \exp(-\lambda o)$，取 $\lambda = 4$（98% 概率 $o=0$，2% 多移除），平滑阶段过渡。
        2.  **优化器状态重置:** 每次新增移除一个 token 时，重置 AdamW 的一阶/二阶矩估计，避免二阶梯度的突变导致训练崩溃。
        3.  **左移除:** 从 CoT 序列**开头**移除 token（右移除显著更差，因为末尾 token 依赖前文，仅靠 answer 前少量位置难以承载）。
    *   *目标:* 全部 CoT token 移除后，模型输入文本直接输出 `Threat/Safe` 与概率，不输出任何中间推理文本。极大缩短推理延迟。
    *   *未来工作 (不在本期范围):* Quiet-STaR (Zelikman et al. 2024) 的 `<|startofthought|>` 元 token + mixing head + REINFORCE 方案更前沿，但其完整版需 8×H100、单卡不可行，列为后续扩展。
    *   *硬件核算 (RTX 4090 24GB):* 8B 参数 4-bit (NF4) 约 5-6GB 显存，加梯度/优化器/LoRA 权重，训练峰值约 14-18GB，**单卡 QLoRA 可行**。但 Stepwise Internalization 每移除一批 token 需再训若干 epoch，总训练工时较长，建议 $\Delta$ 适中（8 起步，必要时下调）并设置早停。

### 3. 评估阶段 (Evaluation Phase: 评估什么，怎么评估，数据是什么)

为了证明模型的有效性并与 Baseline 形成严谨对比，评估必须基于未见过的盲测数据。

*   **盲测数据集组装 (Zero-Knowledge Test Set)**
    * **草垛 (The Haystack):** `data/synthesized/hard_negatives.jsonl`（1231 条由 synthesis 展开的安全对照言论）+ HuggingFace `allenai/WildChat-nontoxic` 英文子集（5000 条真实用户-ChatGPT 多轮对话首条 user turn，见 `scripts/prepare_haystack.py`）。原 `doj_non_criminal.jsonl` 已弃用。
    * **针 (The Needles):** 划分在一开始使用 Teacher LLM 生成的隐式犯罪文本为训练集与测试集、使用测试集中未参与任何训练的隐式犯罪文本。**测试集 needles 必须与训练集零重叠**，且造数 Teacher 与训练 Student 不得共享该测试集。
    * **混合映射:** 通过随机种子脚本混合，生成 `test_blind.csv`，记录每条样本的来源（haystack/needle）与 ground-truth 标签。

*   **评估基线 (Baselines):**
    * **Baseline 1 (判别式模型):** 部署预训练的 `unitary/toxic-bert`，统计其在隐式语境下的漏报情况。
    * **Baseline 2 (通用生成式):** 使用完全未经微调的 `Qwen3-8B`，采用 Zero-shot Prompting，评估其原始推理能力。
    * **Baseline 3 (Ablation - 显式模型):** 使用我们在 Phase 1 训练完成的，未进行思维链内化的显式 CoT 模型。
    * **Baseline 4 (Ablation - 无 DPO):** 使用 Phase 1 SFT 后、未经 Phase 2 DPO 的模型，验证 DPO 对 FPR 的贡献。
    * **Baseline 5 (Ablation - 仅 DPO):** 使用 Phase 2 DPO 后的 LoRA 权重 + SFT 分类头，验证 DPO 在 SFT 基础上对隐式意图识别的边际贡献（与 Baseline 4 对照，量化 DPO 对 TPR/FPR 的双向影响）。
    * **Baseline 6 (本方法 - 隐式内化):** 使用 Phase 3 Stepwise Internalization 训练后的模型，思维链已内化，无显式 CoT 输出。

    > 注：原 Plan 中的 `RoBERTa-Large` 判别基线已移除：本任务以 LLM 内化为主路线，RoBERTa 蒸馏源作为可选增强保留在 Phase 1，但不单独作 baseline。

*   **量化指标矩阵计算 (Quantitative Metrics)**
    * 所有测试必须基于硬性阈值（如判定概率 > 0.5 即视为 Threat）。
    * **FPR (False Positive Rate):** $\frac{FP}{FP + TN}$。反映将 50,000 条安全评论误判的数量（要求趋近于 0）。
    * **TPR (True Positive Rate / Recall):** $\frac{TP}{TP + FN}$。反映找出的隐式威胁数量。
    * **F1-Score:** 综合评估准确与召回。
    * **计算效率指标 (Efficiency):** 对比 步骤 2.2 (显式模型) 与 步骤 2.3 (隐式内化模型) 的推理延迟（Latency）。统计输出每条结果的 **Tokens/Second** 与 **绝对耗时 (ms)**，以证明隐式内化的工程价值。

*   **质性评估依据 (LLM-as-a-Judge)**
    * 采用 Zheng et al. (2023) 方法，在消融实验阶段，提取少量显式推理的输出，利用独立大模型评估其捕捉"上下文异常（Context Anomaly）"的准确度，提供文献支撑所需的质性图表。
    * **一致性指标 (S1/S2):** S1 把 tie + 不一致都算 tie；S2 仅在非 tie 样本上计算一致率。报告 judge-human 一致性，目标 S2 >= 80%。
    * **ToXCL 自定义解释评估 (Hoang et al. 2024 Alg.1):** 对解释输出采用"匹配则计指标、不匹配罚 0、双方均 `[None]` 加分"的算法，惩罚多余解释。
    * **偏差监控:** 报告位置偏差（biased-first 率）、冗长偏差、自我增强偏差（避免同族模型既当系统又当裁判）。
    * **开源微调 judge:** 按 Zheng et al. (2023) App F，微调 Qwen3-8B 三分类 judge 作为可复用评估器，统计一致性 16% -> 65%、格式错误 -> 0% 的提升。

### 4. 成果转化与论文映射 (Course Deliverables，暂时不写，要先写代码)

按照科技文献规范，将上述工程步骤直接映射为学术论文的四大核心部分：

*   **Abstract & Introduction:**
    明确定义 Key Issue（无特征词条件下的隐式意图识别），提出采用 QLoRA 微调与隐式思维链内化技术的本地化小模型方案，替代高成本且存在隐私泄露风险的 API 调用。引用 Wen et al. (2023) 作为痛点支撑。
*   **Methodology:**
    绘制系统架构图。分为三大模块详细论述：
    1. 认知蒸馏与 SFT。
    2. 基于 DPO 的边界偏好对齐原理。
    3. 引用 Deng et al. (2024)，论述如何通过目标截断进行隐式思维链（Implicit CoT）内化，降低推理阶段的计算复杂度（FLOPs）。
*   **Experiments & Results:**
    展示核心数据表。
    *   **表 1:** 盲测数据集上的 TPR、FPR 和 F1-Score 对比（展示你的模型如何碾压 Toxic-BERT）。
    *   **表 2:** 推理延迟对比（Explicit CoT vs Implicit CoT），证明内化后推理速度显著提升，具备边缘设备部署潜力。
    *   **图 1:** 使用混淆矩阵（Confusion Matrix）可视化展示误报率的降低。
*   **Conclusion:**
    总结通过局部权重适配（LoRA）与推理内化，消费级显卡即可完成领域强相关的隐式意图识别。
