# 开发日志 (Develop Log)

## 2026-07-04 08:06 — 项目初始化与代码骨架搭建

### 本次代码更改做了什么
1. **完善 `docs/Plan.md`**：基于 5 篇参考文献的深度阅读，对计划做了 6 处实质性补充：
   - 数据来源澄清：DOJ 新闻稿是"犯罪叙事"而非"隐式意图言论"，需经 Teacher LLM 按语言学特征提示改写；明确正样本种子 / 硬负样本 / 草垛三种角色。
   - Phase 1 SFT：引入 ToXCL 风格的 pooled hidden-state 分类头 + CLM 联合损失 `α·L_cls + β·L_clm`，避免纯生成式误差传播；辅助犯罪类别生成前缀；可选 RoBERTa-Large KL 蒸馏；Conditional Decoding Constraint。
   - Phase 2 DPO：LLM-as-Judge 自动生成偏好对（位置交换一致性过滤 + 参考引导打分 + 三分类偏好方案）；DPO β 起始 0.1；规则检测器奖励整形；judge-human 一致性校验 + 开源微调三分类 judge。
   - Phase 3：明确采用 Stepwise Internalization（Deng 2024）为主方法，含 Removal Smoothing λ=4、优化器重置、左移除、Δ=8；Quiet-STaR 列为未来工作（8×H100 不可行）。
   - 评估：新增 LLM-judge S1/S2、ToXCL 自定义解释评估、偏差监控、开源微调 judge、Baseline 4/5。
   - 硬件核算：RTX 4090 24GB QLoRA 峰值 14-18GB 可行；Stepwise 多轮重训工时长。
2. **建立代码结构目录**：`src/{utils,data,models,training,eval}/` + `configs/` + `scripts/` + `tests/` + `data/{raw,synthesized,preference,blind,cache}/`。
3. **实现全流程模块**（不执行训练，仅写代码）：
   - `src/utils/`：env.py（.env 加载，多 provider）、config.py（YAML+dataclass）、seed.py、logging.py。
   - `src/data/`：llm_client.py（GLM/Gemini/OpenAI 兼容，失败重试降级）、doj_loader.py、synthesis.py（Phase 0 造数）、hard_negatives.py、preference.py（DPO 偏好对 + 位置交换一致性）、dataset.py、blind_set.py。
   - `src/models/`：student.py（QLoRA + ToXCL 分类头 + Conditional Decoding）、classifier_head.py（RoBERTa Teacher）、judge.py（开源三分类 judge + S1/S2 + 偏差统计）。
   - `src/training/`：sft_dataset.py、sft.py、dpo.py、implicit_cot.py（Stepwise Internalization 完整实现：调度 + Removal Smoothing + 优化器重置 + 左移除 + thought span 定位）。
   - `src/eval/`：metrics.py（FPR/TPR/F1/混淆矩阵/延迟）、baselines.py（5 个基线）、llm_judge.py（质量评估 + ToXCL Alg.1）、run_eval.py。
   - `configs/default.yaml` + `scripts/run_*.py`（8 个入口）+ `scripts/run_all.sh`。
   - `tests/test_logic.py`：13 个纯逻辑单测（不依赖 GPU/网络）。
4. **环境配置**：`.env.example`（GLM/Gemini/OpenAI base_url + API_KEY 模板，供用户填写）；`.gitignore`（排除 crawler venv、生成数据、模型权重、node_modules；保留 crawler 源码与源数据 jsonl）。
5. **Git 初始化**：`git init` + 首次 commit（91 文件，20502 行）。

### 开发中遇到的问题与解决方案
- **问题：crawler 目录自带一个 Python venv（lib/bin/include/pyvenv.cfg），与 ML conda 环境是两套东西。**
  - 解决：`.gitignore` 中明确排除 `crawler/{bin,lib,lib64,include,pyvenv.cfg,__pycache__}`，只保留 crawler 的源码（config.py/doj_spider.py/filter_criminal.py/run.py）与 output 源数据。注释中说明二者分离。
- **问题：PDF 在当前环境无法用 read 工具直接解析。**
  - 解决：委派 explore 子代理读取 3 个 PDF（用 pypdf 装在 /tmp 隔离目录，不污染项目与 conda env）；2 个 LaTeX 源直接读取。得到完整技术摘要用于完善 Plan。
- **问题：免费 Teacher LLM（GLM-4-Flash / Gemini-2.0-Flash）质量弱于 GPT-4，直接用作 DPO judge 风险高。**
  - 解决：在 Plan 与 preference.py 中加入 (a) 位置交换一致性过滤、(b) 参考引导打分、(c) judge-human 一致性校验（目标 S2≥80%）、(d) 按 Zheng 2023 App F 微调开源三分类 judge 作为廉价补充。
- **问题：Stepwise Internalization 训练成本高，RTX 4090 单卡多轮重训工时长。**
  - 解决：Plan 中建议 Δ 起始 8、必要时下调、设置早停；implicit_cot.py 实现了优化器重置与 Removal Smoothing 以稳态训练。
- **问题：未运行训练前需保证纯逻辑可测，但 ML env 无 pytest。**
  - 解决：tests 已写好但未执行（避免随意装包违反 AGENTS.md 约定）；待用户确认后再 `conda install pytest` 跑测试。

### 代码实现要点
- `src/data/llm_client.py`：统一 `chat(messages, ...)` 接口，`build_client(provider_name)` 按 .env 自动选择可用 provider，`safe_json_extract` 从 LLM 输出抽取 JSON。
- `src/training/implicit_cot.py`：核心调度 `s(t)=⌊Δ·t/T⌋` + `removal_smoothing_offset(λ=4)` + `apply_removal(left=True)` + `reset_optimizer`；`_locate_thought_span` 通过 `<thought>`/`</thought>` token 定位可移除段。
- `src/models/student.py`：`StudentModel` 封装 QLoRA 基座 + `ClassifierHead`（mean-pool + linear），forward 返回 `{logits, cls_logits, clm_loss, cls_loss}`，`save`/`load` 同时保存 LoRA adapter 与分类头权重。
- `src/eval/run_eval.py`：统一 `Baseline.predict -> Prediction(label, prob, cot, latency_ms, tokens)`，输出 `predictions_<name>.json` + `metrics_table.md`。

### 测试结果
- 未运行（ML env 缺 pytest，且按 AGENTS.md 约定不随意装包）。
- `tests/test_logic.py` 覆盖：metrics、removal schedule/smoothing/apply、label 转换、ToXCL 解释评分、judge S1/S2、safe_json_extract、doj_loader、env 加载、config roundtrip。待 `conda install pytest` 后执行 `python -m pytest tests/test_logic.py -q`。

### 性能优化（预留）
- StudentModel 启用 gradient_checkpointing；QLoRA 4-bit NF4 + bf16 compute dtype 适配 RTX 4090。
- Stepwise Internalization 中 thought span 预计算一次复用，避免每 step 重定位。
- LLM 客户端指数退避重试（3 次），造数过程中每 50 条 flush 一次避免中断丢失。

### 下一步
- 用户配置 `.env`（填入 GLM/Gemini API_KEY 与 base_url）。
- `conda install pytest` 后跑单测验证逻辑层。
- 小规模造数（`--limit 50`）验证 Teacher LLM prompt 与解析链路。
- 之后按 `scripts/run_all.sh` 各阶段顺序推进。
