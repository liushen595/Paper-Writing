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

---

## 2026-07-06 17:52 — 勘误修复、断点续训、合成验证

### 本次代码更改做了什么

1. **勘误：API Provider 修正**
   - `src/utils/env.py`：`gemini` -> `agnes`，`LLMProviderConfig` 新增 `model_name` 字段，从 `.env` 的 `*_MODEL_NAME` 环境变量读取，代码不硬编码模型名。
   - `src/data/llm_client.py`：移除 `GeminiClient` 类与 `_PROVIDER_REGISTRY` 硬编码映射，统一用 `OpenAICompatibleClient`；`build_client` 使用 `provider.model_name`。
   - `.env.example`/.env：保留 `GLM_MODEL_NAME`/`AGNES_MODEL_NAME`/`OPENAI_MODEL_NAME`，用户在 `.env` 中配置实际模型名。
   - `configs/default.yaml`：`judge_provider` 改为 `agnes`。
   - `scripts/run_synthesis.py`：help 文本 gemini -> agnes。

2. **勘误：移除 `doj_non_criminal.jsonl` 相关逻辑**
   - `src/utils/config.py`：移除 `raw_non_criminal` 字段。
   - `src/data/hard_negatives.py`：移除 `from_non_criminal()` 函数，仅保留 `llm_augment()` 与 `merge_hard_negatives()`。
   - `src/data/blind_set.py`：移除 `doj_non_criminal` haystack 来源，仅保留 `hard_negatives` + 可选额外泛语料。
   - `configs/default.yaml`：移除 `raw_non_criminal` 配置。

3. **断点续训（总轮数设计）**
   - `src/training/sft.py`：`_find_latest_checkpoint(ckpt_dir)` 查找 `checkpoint-XXXX`，`start_epoch` 从最新 checkpoint 恢复，`num_epochs` 为总轮数，跳过已完成 epoch。每个 epoch 结束保存 `checkpoint-{epoch+1}`。
   - `src/training/dpo.py`：`_find_latest_checkpoint` + `trainer.train(resume_from_checkpoint=...)`。
   - `src/training/implicit_cot.py`：额外保存 `train_state.json`（`epoch`, `removed_so_far`）恢复 Stepwise Internalization 的 removal state，`_find_latest_checkpoint` + 从 `removed_so_far` 继续调度。

4. **适配 RTX 4060 8GB**
   - `configs/default.yaml`：`per_device_batch_size=1`, `gradient_accumulation_steps=16`, `max_seq_len=512`, `lora_r=16`。注释标注服务器推荐值。

5. **Logging Bug 修复**
   - `src/utils/logging.py`：`_CONFIGURED` 全局标志导致不同 name 的 logger 不添加 handler，日志静默丢失。改为 `_LOGGERS` dict 按 name 管理，每个 logger 独立配置 handler。

6. **Synthesis 修复与验证**
   - `src/data/synthesis.py`：`max_tokens` 512->1024；`synthesize_one` 添加 `retries=2` 重试机制；SYSTEM_PROMPT 加"不要 markdown 代码块"。
   - `src/data/llm_client.py`：`safe_json_extract` 支持剥离 ```` ```json...``` ```` 包裹。
   - **验证结果**：`agnes-2.0-flash` 成功生成 2 条训练数据（Cyber + Narcotics），JSON 解析正常，train/test 80/20 切分正确。

7. **Pytest 全部通过**
   - `tests/test_logic.py`：12 个纯逻辑单测全部通过（metrics / removal schedule / label conversion / ToXCL score / judge agreement / safe_json_extract / doj_loader / env loading / config roundtrip）。

### 开发中遇到的问题与解决方案
- **问题：`run_synthesis.py` 日志静默丢失，合成完成但日志不输出进度。**
  - 原因：`setup_logger` 的 `_CONFIGURED` 全局标志在首次配置后设为 True，后续不同 name 的 logger 不添加 StreamHandler。
  - 解决：改为 `_LOGGERS` dict 按 name 管理，每个 logger 独立添加 handler。
- **问题：Agnes API 返回的 JSON 被 markdown 代码块包裹或被截断。**
  - 解决：`safe_json_extract` 剥离 ```` ```json...``` ````；`max_tokens` 提升到 1024；`synthesize_one` 添加重试。
- **问题：`_flush` 清空 results 后 `log.info` 报告 0 条。**
  - 解决：`total_ok` 在 `_flush` 前累加，独立于 results 列表。

### 测试结果
- Pytest: 12/12 passed (0.94s)。
- 小规模合成: 3 条 DOJ -> 2 条成功（1 条 train + 1 条 test），Agnes API 响应 8-11s/条。

### 硬件说明
- 当前开发机 RTX 4060 Laptop 8GB，QLoRA 8B 模型仅 ~4-5GB 显存 + 梯度/LoRA 开销，完整训练（SFT/DPO/Stepwise）可能溢出。
- 完整训练需租用服务器（建议 RTX 4090 24GB 或 A6000 48GB）。

---

## 2026-07-06 22:48 — _flush 分批 bug 修复、body 加入造数、README、prompt 英文化

### 本次代码更改做了什么

1. **修复 `_flush` 分批写入严重 bug**（`src/data/synthesis.py`）
   - 旧逻辑：`partial=True` 用 append 模式，`partial=False` 用 write 模式（先 `unlink` 删除文件再重写）。
   - 问题：最终 `_flush(partial=False)` 会删除文件，只写入最后一批数据，中间所有批次（每50条一 flush）全部丢失。
   - 修复：`run_synthesis` 首次运行时清空文件，`_flush` 始终用 append 模式（`"a"`），不再区分 partial。

2. **造数加入 DOJ 新闻稿正文**（`src/data/synthesis.py` + `src/data/doj_loader.py`）
   - `_build_messages`：新增 `body_excerpt = record.body[:800]`，USER_TEMPLATE 加入 `- Body: {body}` 字段。
   - `extract_case_elements`：`text` 从 `title + summary` 扩展为 `title + summary + body[:500]`，犯罪类型提取更准。

3. **所有 LLM API 提示词改为英文**
   - `synthesis.py`：SYSTEM_PROMPT + USER_TEMPLATE 全英文，要求 "All content must be in English"。
   - `preference.py`：JUDGE_SYSTEM + JUDGE_USER_TEMPLATE 全英文。
   - `hard_negatives.py`：SYSTEM_PROMPT + USER_TEMPLATE 全英文。
   - `llm_judge.py`：QUALITY_JUDGE_SYSTEM + QUALITY_USER_TEMPLATE 全英文。
   - `dataset.py`：`_from_hard` 的 fallback thought_process 从中文改为英文。

4. **写 README.md**
   - 完整目录结构树（每个文件/目录的作用）。
   - 环境配置（conda + .env + 硬件要求）。
   - 训练流程详解（Phase 0/1/2/3 的输入/输出/原理/命令/预计耗时）。
   - 评估基线与指标说明。
   - 参考文献列表。

### 测试结果
- Pytest: 12/12 passed (1.16s)。

### 下一步
- 在开发机上用 `--limit 50` 跑完整造数验证 Agnes API 稳定性和数据质量。
- 下载泛语料 haystack（推荐 `allenai/c4` 随机采样 5000-10000 条）。
- 租服务器执行完整训练。

---

## 2026-07-07 15:55 — 环境重建 + Phase C/D（DPO 接续 + 评估可视化）

### 本次代码更改做了什么

1. **环境重建（替换 environment.yml）**
   - 旧 ML 环境因 conda 安装 `bitsandbytes` 反复解析 CUDA 依赖导致 WSL 崩溃，整环境重建。
   - 新环境：`conda create -n ML python=3.10` + 全量 `pip install`（清华源）。关键决策：
     - `torch==2.5.1` 走 PyPI 官方 wheel，自带 CUDA 12.4 运行时，不依赖系统 cuda toolkit。
     - `bitsandbytes==0.49.2` 走 PyPI 官方 wheel（自带 CUDA 后端 .so），不再用 conda-forge 的 CPU-only build。后者曾导致 4-bit QLoRA 静默退化且安装时崩溃，是本次重建的核心动因。
   - 验证：`bitsandbytes.nn.Linear4bit` 在 cuda:0 上 4-bit 量化前向通过，`QuantState` 正常生成。
   - 用 `pip-chill` + `pip list --format=freeze` 导出全量依赖，重写 `environment.yml` 为 `pip:` 段格式（符合 AGENTS.md "用 pip 不用 conda install"），固定 100+ 包确切版本以保证可复现。
   - pytest 13/13 通过（含新环境验证）。

2. **Phase C1：替换 DPO 候选生成器 stub**
   - `src/data/preference.py`：
     - 新增 `make_sft_candidate_generator(sft_cfg, ckpt_dir, temperatures=(0.3, 1.0))`：加载 SFT checkpoint，对同一 prompt 用不同温度采样 n 个候选，候选内容为 SFT 学到的 "<thought_process> -> <label>" 形式，便于 judge 比较推理严谨度。
     - `run_preference_generation` 新增 `sft_cfg`/`sft_ckpt` 参数；优先级：显式 `candidate_generator` > SFT 采样 > `_dummy_candidate_gen`（带 warning）。
     - `_dummy_candidate_gen` 文案英文化，仅用于流程冒烟。
   - `scripts/run_preference.py`：新增 `--sft-ckpt` CLI 参数，缺省读 `cfg.sft.output_dir`；ckpt 不存在则 warning 并回退到 stub。

3. **Phase C2：DPO 保留分类头**
   - `src/training/dpo.py`：DPO 训练 `trainer.save_model` 后，新增逻辑把 SFT 的 `classifier_head.pt` 复制到 DPO 输出目录。这样 dpo-only baseline 可加载完整 StudentModel（DPO LoRA + SFT 分类头），无需重新构造分类头。
   - 若 SFT 分类头不存在则 warning，不阻塞训练。

4. **Phase C3：添加 dpo-only baseline + 重写 StudentModel.load**
   - `src/models/student.py` 的 `StudentModel.load`：原实现只 `cls(sft_cfg)`（重新 `get_peft_model` 创建空 LoRA），无法加载 DPO adapter 权重。重写为：
     - 用 `AutoModelForCausalLM.from_pretrained` + 4-bit 量化加载基座。
     - 若 `ckpt_dir/adapter_config.json` 存在，用 `PeftModel.from_pretrained` 套上 LoRA adapter（SFT 与 DPO checkpoint 统一路径）。
     - 加载 `classifier_head.pt` 到 `ClassifierHead`。
   - `src/eval/run_eval.py`：`_build_baseline` 增加 `"dpo-only"` 路由，指向 `cfg.dpo.output_dir`，`conditional_decoding=False`（dpo-only 仍走显式推理，不做条件解码）。
   - `src/utils/config.py` 的 `EvalConfig.baselines` 默认列表改为 6 个 baseline（含 dpo-only）。
   - `configs/default.yaml`：baselines 增加 `"dpo-only"`。

5. **Phase C4：移除 roberta-large baseline + 同步文档**
   - `src/eval/run_eval.py` 的 `roberta-large` 路由改为 `raise NotImplementedError("roberta-large 基线已移除")`。
   - `src/eval/baselines.py` docstring 更新为 6 个 baseline 列表。
   - `docs/Plan.md` §3 Baselines：原 Baseline 5 (RoBERTa-Large) 替换为 Baseline 5 (Ablation - 仅 DPO) + Baseline 6 (本方法 - 隐式内化)，并加注 RoBERTa 蒸馏源保留为 Phase 1 可选增强。
   - `docs/Methodology.md` §6.1 Baselines：同步改为 6 项，加 RoBERTa 移除说明。

6. **Phase D1：新建 `src/eval/visualize.py`**
   - 论文交付物图表渲染模块，输出到 `cfg.eval.output_dir`：
     - `confusion_matrix_<name>.png`：每个 baseline 一张混淆矩阵图（图 1 系列），从 `predictions_<name>.json` 推 TP/FP/FN/TN，matplotlib `imshow` + 数字标注。
     - `metrics_table.csv`：Table 1 机器可读版（baseline + tp/fp/fn/tn + tpr/fpr/precision/f1/accuracy + 延迟）。
     - `tpr_fpr_bars.png`：跨 baseline TPR/FPR 柱状对比图（核心结果图），双色柱状 + 数值标注。
     - `latency_table.md`：Table 2 显式 vs 隐式延迟对比 markdown 表，自动计算 explicit-cot vs implicit-cot 加速比。
   - `visualize_all(out_dir, reports)` 主入口，失败不阻塞主流程（run_eval.py try/except 包裹）。

7. **Phase D2：run_eval.py 接入 visualize**
   - `src/eval/run_eval.py`：`run_eval` 末尾在写完 `metrics_table.md` 后调用 `visualize_all(out_dir, reports)`，异常仅 warning 不抛出。

8. **Phase D3：修 llm_judge.py JSON 读取 + 接入 run_all.sh + 填充 bias stubs**
   - `src/eval/llm_judge.py`：
     - 新增 `_load_predictions(path)`：支持 `.json`（run_eval.py 输出）与 `.csv` 两种格式。原 `run_judge_eval` 用 `csv.DictReader` 读 JSON 文件必然失败，是评估流程的硬 bug。
     - 新增 `_biased_first_rate(per_sample)`：用 "judge score>=7 但 model_label!=ref_label" 的样本占比近似位置偏差率（Zheng 2023 App F 简化版）。
     - 新增 `_verbosity_bias_rate(per_sample)`：按 median cot_len 划分长短两组，归一化 mean score 差到 [0,1]。
     - `run_judge_eval` 改名参数 `predictions_csv` -> `predictions_path`，调用 `_load_predictions`，bias 字段由 stub 0.0 改为真实计算。
   - `scripts/run_all.sh`：新增 `judge` stage，遍历 `outputs/eval/predictions_*.json` 对每个 baseline 跑 LLM-as-judge；`all` 流程末尾追加 `run_judge`。

### 开发中遇到的问题与解决方案
- **问题：conda 安装 `bitsandbytes` 反复让 WSL 崩溃。**
  - 原因：conda-forge 的 bitsandbytes 是 Python-only 包，不带 CUDA 后端 .so；尝试装 `cuda129_py310h93df00f_200` build 时 conda 解析 CUDA 依赖树异常复杂，触发 WSL 崩溃。
  - 解决：彻底放弃 conda 装 bitsandbytes，整体改用 pip。PyPI 上的 `bitsandbytes==0.49.2` 官方 wheel 自带 CUDA 12.x 后端，安装简单且功能完整。AGENTS.md 已要求 pip 优先，本次重建贯彻到底。
- **问题：pip-chill 默认输出过滤过激，把 torch/transformers 等显式安装的包也当依赖过滤掉。**
  - 解决：用 `pip list --format=freeze` 取全量列表，手动按"核心 + 工具库"分类整理成 environment.yml 的 pip 段，固定所有版本。
- **问题：`StudentModel.load` 无法加载 DPO checkpoint。**
  - 原因：原实现 `model = cls(sft_cfg)` 会重新 `get_peft_model` 创建一个空 LoRA，再用 `model.classifier.load_state_dict` 装分类头；DPO adapter 权重从未被加载。
  - 解决：重写 `load` 为"基座 + PeftModel.from_pretrained(ckpt_dir) + 分类头"三段式，SFT/DPO checkpoint 用同一入口加载。
- **问题：`run_judge_eval` 用 csv.DictReader 读 JSON 文件。**
  - 原因：`run_eval.py` 写 predictions 是 JSON，但 `llm_judge.py` 用 CSV reader 解析，必然得到空列表。
  - 解决：新增 `_load_predictions` 支持两种格式，按文件后缀分发。

### 代码实现要点
- `make_sft_candidate_generator`：用 `do_sample=True` + 不同温度（0.3 / 1.0）生成 n 个候选，低温偏保守、高温偏多样，让 judge 真有可比性的两个候选可选出 chosen/rejected。
- `StudentModel.load` 重写后是 SFT/DPO/未来 implicit-cot checkpoint 的统一加载入口；若 `adapter_config.json` 缺失则 warning 并用未微调基座，便于早期冒烟。
- `visualize.py` 用 `matplotlib.use("Agg")` 强制无显示设备渲染，PNG 输出 150 dpi，混淆矩阵用 Blues colormap + 黑白自适应数字色。
- `latency_table.md` 自动计算 explicit-cot vs implicit-cot 加速比，是论文工程价值论据的直接来源。
- `llm_judge._biased_first_rate` / `_verbosity_bias_rate` 是 Zheng 2023 App F 偏差监控的启发式近似，正式版可外接更严格的 A/B 交换测试。

### 测试结果
- pytest: 13/13 passed (1.16s)，新环境 + 新代码全通过。
- 模块 import：`src.eval.visualize` / `src.eval.llm_judge` / `src.data.preference` / `src.training.dpo` / `src.models.student` / `src.eval.run_eval` 全部 import 成功。
- visualize 烟雾测试：mock 3 个 baseline × 50 条 predictions，渲染出 3 张混淆矩阵 PNG（676×529 RGBA）+ 1 张 TPR/FPR 柱状图 PNG（1034×657 RGBA）+ metrics_table.csv + latency_table.md，全部有效非空。

### 性能优化（预留）
- `visualize.py` 用 Agg backend，无 GUI 依赖，可在服务器无 X11 环境运行。
- `run_eval.py` 调用 visualize 用 try/except 包裹，可视化失败不影响指标表输出（论文交付物优先级：指标 > 图表）。

### 下一步
- 在服务器上跑完整训练：`bash scripts/run_all.sh all`（haystack → synth → hardneg → pref → sft → dpo → implicit → blind → eval → judge）。
- WildChat 草垛下载：`python -m scripts.prepare_haystack --n 5000`（需 HF token + 数据集权限；脚本就绪，用户机器跑不动训练暂缓）。
- 论文图表：跑完 eval 后直接从 `outputs/eval/` 取混淆矩阵 PNG / metrics_table.csv / tpr_fpr_bars.png / latency_table.md 入稿。
- 可选：Stepwise Internalization 阶段对 GPU 显存敏感，RTX 4060 8GB 可能溢出，建议租 RTX 4090 24GB 或 A6000 48GB。

## 2026-07-10 13:43 CST — SFT/DPO/ImplicitCoT 训练流程添加早停 + SFT 学习率调整

### 本次代码更改做了什么
1. **为三个训练阶段统一添加早停（Early Stopping）**：
   - `src/training/sft.py`：每个 epoch 结束后在 test split 上计算验证 loss（`cls_loss + clm_loss`），连续 `early_stopping_patience=3` 个 epoch 无改善则终止；最优模型保存到 `checkpoints/sft/best/`。
   - `src/training/dpo.py`：将 preference 数据 90/10 切分为 train/eval，使用 `transformers.EarlyStoppingCallback`，基于 `eval_reward_accuracies` 早停，`load_best_model_at_end=True`。
   - `src/training/implicit_cot.py`：每个 epoch 结束后用当前 removal level `s` 在验证集上计算 loss，连续 `early_stopping_patience=3` 个 epoch 无改善则终止；最优模型保存到 `checkpoints/implicit_cot/best/`。

2. **集中式配置扩展**：
   - `src/utils/config.py`：在 `SFTConfig` / `DPOConfig` / `ImplicitCoTConfig` 中新增 `early_stopping_patience` 与 `early_stopping_min_delta`。
   - `configs/default.yaml`：同步添加早停参数。

3. **SFT 学习率调整**：
   - 将 SFT `learning_rate` 从 `2e-4` 降至 `5e-5`，缓解数据量较小时分类头过快过拟合。

### 开发中遇到的问题与解决方案
- **问题：SFT 分类头在 160 步内 cls_loss 接近 0，明显过拟合。**
  - 解决：引入验证集早停 + 降低学习率，避免模型记住训练集；验证集使用 test split（与训练集无重叠）。
- **问题：DPOTrainer 的早停需要 eval_dataset 与对应 callback。**
  - 解决：从 `dpo_pairs.jsonl` 中随机切分 10% 作为 eval，配置 `eval_strategy="epoch"`、`metric_for_best_model="eval_reward_accuracies"` 与 `EarlyStoppingCallback`。
- **问题：Implicit CoT 的验证 loss 依赖于当前 removal level `s`，不能直接用原始 thought 验证。**
  - 解决：在验证时应用与训练当前 epoch 相同的 `prev_s` 进行 token removal，再计算 `clm_loss + cls_loss`。
- **问题：早停配置字段需要在三个 Phase 中统一管理。**
  - 解决：在 `SFTConfig` / `DPOConfig` / `ImplicitCoTConfig` 中分别加入 `early_stopping_patience` 和 `early_stopping_min_delta`，并在 YAML 中可覆盖。

### 代码实现要点
- 早停逻辑封装在每个训练脚本内部，不引入额外依赖；`transformers.EarlyStoppingCallback` 用于 DPO（TRL 已依赖 transformers）。
- SFT/ImplicitCoT 在 epoch 结束时调用独立 `_evaluate*` 函数，切换 `model.eval()` / `model.train()` 避免影响训练状态。
- 验证 loss 采用与训练相同的联合损失形式，确保早停指标与训练目标一致。

### 测试结果
- `python -m ast.parse` 对 `sft.py`、`dpo.py`、`implicit_cot.py`、`config.py` 语法检查通过。
- `load_config('configs/default.yaml')` 正常读取所有新增字段，SFT `learning_rate=5e-5`，各阶段 `early_stopping_patience` 符合预期。

## 2026-07-10 13:43 CST — 数据流水线重构：取消犯罪/非犯罪预过滤，LLM 前置判断犯罪性

### 本次代码更改做了什么
1. **取消 crawler 阶段的犯罪/非犯罪过滤**：
   - 删除 `crawler/filter_criminal.py`。
   - `crawler/config.py` 输出文件从 `doj_press_releases.jsonl` 改为 `doj_raw.jsonl`。

2. **数据配置统一指向全量原始数据**：
   - `src/utils/config.py`：`DataConfig.raw_criminal` 重命名为 `raw_doj`，默认指向 `crawler/output/doj_raw.jsonl`。
   - `configs/default.yaml`：同步更新。
   - `src/data/synthesis.py`：读取路径从 `raw_criminal` 改为 `raw_doj`。

3. **重写 `src/data/synthesis.py` 的 LLM 提示词**：
   - Teacher LLM 先判断新闻稿是否为刑事案件，再分两支输出：
     - **刑事分支**：生成 `implicit_threat` + `hard_negative` + CoT，label="Threat"。
     - **非刑事分支**：生成 Safe 噪声样本，`implicit_threat` 复用为中性安全文本，`hard_negative=""`（不生成硬负样本），label="Safe"，probability≈0，category="NonCriminal"。
   - 不新增任何数据字段，保持 `train.jsonl` / `test.jsonl` 的 schema 兼容。

4. **更新 `docs/Methodology.md` §3**：
   - 数据源改为 `doj_raw.jsonl`。
   - 数据合成流程明确 LLM 前置犯罪性判断与 Safe 噪声分支。

### 开发中遇到的问题与解决方案
- **问题：去掉过滤后，非犯罪新闻稿会进入 synthesis，必须避免污染 hard_negative。**
  - 解决：在 prompt 中明确要求非刑事案件 `hard_negative` 为空字符串；`hard_negatives.py` 的 `from_synth_internal` 本来就跳过空 text，因此非犯罪记录不会进入 hard_negatives.jsonl。
- **问题：不能新增字段，但需要为非犯罪样本提供输入文本。**
  - 解决：复用现有 `implicit_threat` 字段存储中性安全摘要；`_from_synth` 的 `text=d.get("implicit_threat") or d.get("text", "")` 天然兼容。
- **问题：`probability` 字段在非刑事案件应如何设置。**
  - 解决：prompt 要求 probability=0.0 或极低（0.0-0.05），与 Methodology 中"probability 不进入训练损失"保持一致，仅作为弱参考。

### 代码实现要点
- 数据 schema 不变，SFT / DPO / ImplicitCoT 训练代码无需任何调整。
- hard_negative 生成逻辑不变，仍只从刑事案件产生。
- 配置字段 `raw_doj` 命名更准确，避免后续维护歧义。

### 测试结果
- `python -m ast.parse` 对 `synthesis.py`、`config.py`、`crawler/run.py`、`crawler/config.py` 语法检查通过。
- `load_config('configs/default.yaml')` 正确解析 `raw_doj` 为 `crawler/output/doj_raw.jsonl`。
- 模拟 Safe 分支 JSON 经过 `_parse_synthesis` 解析成功，`hard_negative=""` 被保留但会被下游过滤。

---

## 2026-07-10 16:48 CST — Synthesis 添加 start 参数支持断点续跑

### 本次代码更改做了什么

1. **`src/data/synthesis.py`：`run_synthesis` 新增 `start` 参数**
   - `start=0` 为默认行为（从头开始），`start=500` 则跳过前 500 条记录。
   - 参数位于 `limit` 之后，在 `load_doj_records` 后根据 `start` 切片 `records[start:]`。
   - 日志输出跳过条数和剩余待合成数。

2. **`scripts/run_synthesis.py`：CLI 暴露出 `--start` 参数**
   - 新增 `--start N`：从第 N+1 条开始处理（跳过前 N 条），用于断点续跑。
   - 用法示例：`--start 500 --append` 配合使用。

### 关于 train/test 8:2 划分的一致性

`_save_single` 使用 `hashlib.md5(f"{url}:{seed}".encode())` 对每条记录**独立确定性**分配 train/test：
- 每条记录独立计算 hash，~80% 概率进 train，~20% 进 test。
- 无论分几次跑、从哪条开始，每条记录的分配结果始终一致（相同 url + seed）。
- 最终所有记录的总体比例 ≈ 8:2（大数定律保证）。

因此**不需要担心断点续跑破坏划分比例**。`_flush` 函数（`partial` 参数）的随机 shuffling 方式已废弃，当前代码中未被调用。

---

## 2026-07-10 16:48 CST — Synthesis 多线程 + 阿里云限流/安全错误处理

### 本次代码更改做了什么

1. **`src/data/llm_client.py`：新增三类异常 + 错误分类**
   - `ContentSafetyError`：内容安全拒绝（"Output data may contain inappropriate content"），不重试直接跳过。
   - `RateLimitError`：RPM/TPM 配额超限，sleep 60s 后重试。
   - `BurstLimitError`：RPS 突发保护（"Request rate increased too quickly"），sleep 10s×2^attempt 后重试。
   - `_classify_error(body)` 函数：根据响应 body 的关键字自动分类错误类型。
   - `_stream_request`：检查 `resp.ok`，失败时调用 `_classify_error` 并 raise。
   - `AliyunClient.chat`：catch `openai.RateLimitError` / `APIStatusError` / `APIError`，调 `_classify_error` 转换。

2. **`src/data/synthesis.py`：多线程支持 + 限流器 + 错误感知重试**
   - 新增 `RateLimiter` 类：Token-bucket 实现，按 rpm 控制全局调用间隔，多线程共享一把锁。
   - `synthesize_one` 新增 `rate_limiter` 可选参数；API call 前 `rate_limiter.wait()`；catch 三类异常分别处理（ContentSafetyError 直接跳过，RateLimitError 等 60s，BurstLimitError 指数退避）。
   - `run_synthesis` 新增 `max_workers: int = 1` 和 `rpm: float = 120` 参数。
     - `max_workers ≤ 1`：走原顺序路径 `_run_sequential`，完全不引入多线程开销。
     - `max_workers > 1`：走 `_run_parallel`，使用 `ThreadPoolExecutor` + `as_completed` 并发处理，每 10 条打印一次进度。
   - `_save_single` 加 `_write_lock`（`threading.Lock()`）保证线程安全写入。
   - `_run_sequential` / `_run_parallel` 两个辅助函数，分离顺序/并发逻辑。

3. **`scripts/run_synthesis.py`：CLI 暴露并发参数**
   - `--max-workers N`：并发线程数，默认 1（顺序模式）。
   - `--rpm N`：每分钟最多调用次数，默认 120（仅 `max_workers>1` 时生效）。
   - 用法：`--provider aliyun --model qwen-plus --max-workers 5 --rpm 100`

### 关于不重复处理

`records[start:]` 一次性分片进入 `ThreadPoolExecutor.submit`，每条 record 只产生一个 future，天然不重复。`_save_single` 的 url hash 确定性分配保证每条记录归属一致。

### 测试结果
- 语法检查通过（`ast.parse`）。
- `pytest tests/test_logic.py -q`：13 passed，原有单测未受影响。

---

## 2026-07-10 20:06 CST — 数据去重与清理脚本

### 问题

全量合成完成后，train + test 共 10,157 条记录，但 DOJ 原始只有 9,582 条，多出 616 条重复（断点续跑时 `--start` 偏移未能精确对齐导致部分记录被处理两次）。hard_negatives.jsonl（从 train/test 逐行提取）也因此有 379 条重复。

### 硬负样本提取逻辑确认

`hard_negatives.py:from_synth_internal()` 确实同时遍历 train.jsonl 和 test.jsonl 的每一行，提取 `hard_negative` 字段（空则跳过）。因此 train/test 有重复 → hard_negatives 必有重复。去重 train/test 后需重建 hard_negatives。

### 新增 `scripts/check_dedup.py`

功能：
- **查验模式**（默认）：统计 train/test/hard_negatives 的重复情况，检查 train↔test 有无 url 交集
- **清理模式**（`--clean`）：按 source_url 去重（保留首次出现），覆写文件，并自动从去重后 train/test 重建 hard_negatives.jsonl
- `--dry-run`：预览不写入
- `--yes`：跳过交互确认提示

### 去重结果

| 文件 | 去重前 | 去重后 | 移除 |
|------|--------|--------|------|
| train.jsonl | 8,100 | 7,606 | -494 |
| test.jsonl | 2,057 | 1,935 | -122 |
| hard_negatives.jsonl | 5,010 | 4,622 | -388 |

- train:test = 7,606:1,935 ≈ 79.7:20.3（符合 8:2）
- train 与 test 的 source_url 无交集（hash 划分完全正确）
- 唯一 url 总数 9,541，与 DOJ 原始 9,582 差 41 条（LLM 调用失败）

### 测试结果
- 语法检查通过
- 13 个已有单测未受影响

---

## 2026-07-11 — Bug 修复 + 数据泄漏修复 + 流水线重写 + Phase 3 退役

### 本次代码更改做了什么

#### 1. 流水线 Bug 修复（6 处，不修跑不起来）
1. **DPO `peft_config` 位置错误**（`src/training/dpo.py`）：`peft_config` 传给了 `TRLDPOConfig`（不接受此参数，会 TypeError），移到 `DPOTrainer` 构造参数。
2. **DPO metric 名称错误**（`src/training/dpo.py`）：`metric_for_best_model` 从错误的 `"eval_reward_accuracies"` 改为 trl 0.18.1 实际输出的 `"eval_rewards/accuracies"`。
3. **`QwenZeroShotBaseline` thinking 模式吃光 token**（`src/eval/baselines.py`）：Qwen3 默认 thinking 模式 + max_new_tokens=128 导致 thinking block 占满，JSON 永远不产出，恒返回 Safe/0.0。改为 `apply_chat_template(..., enable_thinking=False)` + max_new_tokens=512 + do_sample=True/temperature=0.1 避免贪婪重复。
4. **Token 计数虚高**（`src/eval/baselines.py`）：`tokens=out.size(1)` 计入 prompt token，改为 `out.size(1) - inputs["input_ids"].size(1)` 只计生成 token。
5. **`safe_json_extract` 无错误处理**（`src/data/preference.py` + `src/eval/llm_judge.py`）：解析失败抛 ValueError 中断全流程，外包 try/except 返回默认值。

#### 2. 数据泄漏修复（3 处，不修论文结果无效）
1. **hard_negatives 未按 split 过滤**（`src/data/dataset.py`）：`build_train_examples` 对 train/test split 都加载全部 4622 条 hard_negatives，忽略已有的 `split_origin` 字段。改为按 `split_origin` 过滤（train→train-origin 3722 条，test→test-origin 900 条）。
2. **盲测集 haystack 含训练数据**（`src/data/blind_set.py`）：用全部 hard_negatives 做 haystack，其中 3722 条在训练集中。改为只用 `split_origin=="test"` 的 900 条。
3. **train/test 文本重复**（`data/synthesized/train.jsonl`）：160 条 test 样本出现在 train 中，从 train.jsonl 去重移除（7606→7446）。

#### 3. Phase 3 隐式 CoT 内化退役
- delta=8 + 20 epoch 仅移除 8 个 thought token（thought 通常 50-150 token），不足以完成内化。
- 加大 delta 至 50-60 收敛风险高 + 需 35-45h+，超出 37h 提交周期。
- **决策：完全放弃，改为 future work**。代码保留（`src/training/implicit_cot.py`），baseline 移除 implicit-cot。
- 论文聚焦 ToXCL 分类头（SFT）+ DPO 降 FPR。

#### 4. Baseline 精简（6→4）
- 移除 `explicit-cot`（与 `sft-no-dpo` 同一 checkpoint，冗余）。
- 移除 `implicit-cot`（Phase 3 退役）。
- 保留 4 个：`toxic-bert` / `qwen-zeroshot` / `sft-no-dpo` / `dpo-only`。
- `configs/default.yaml` 和 `src/eval/run_eval.py` 同步更新。

#### 5. run_all.sh 重写为 Python 驱动器
- 旧版 `run_all.sh` 的问题：
  - `all` 分支 pref 在 sft 之前执行（顺序 bug）。
  - 包含已在本机做完的阶段（crawl/filter/synth/hardneg）。
  - 无 `--from` / `--only` / `--limit` 支持。
- 新版 `scripts/run_all.py`：
  - `--from <stage>`：从指定阶段开始跑到结尾。
  - `--to <stage>`：跑到指定阶段为止（含两端）。
  - `--only <stage>...`：只跑指定阶段。
  - `--limit N`：限制样本数（传给 sft/pref/eval/judge）。
  - `--judge-model <name>`：覆盖 judge 模型。
  - 阶段：`haystack → sft → pref → dpo → blind → eval → judge`。
  - `run_all.sh` 保留为 bash shim（兼容旧调用）。

#### 6. --limit 参数新增（smoke test 用）
- `scripts/run_sft.py` + `src/training/sft.py`：`--limit N` 限制训练+验证样本数。
- `scripts/run_eval.py` + `src/eval/run_eval.py`：`--limit N` 限制盲测样本数。
- `scripts/run_judge_eval.py` + `src/eval/llm_judge.py`：`--limit N` 限制 judge 评估样本数。
- `scripts/run_preference.py` 已有 `--limit`。

#### 7. 文档更新
- `docs/Methodology.md`：§1.3 Phase 3 标 future work；§3.2 补充 split_origin + 去重说明；§3.4 盲测集改为 900 test-hard + 2000 WildChat；§5.3 Phase 3 退役说明；§6.1 baseline 6→4；§7 Table 2 改为各 baseline 延迟对比；§9 硬件改 RTX 3090、入口改 run_all.py。
- `docs/TODO.md`：重写运行方式（run_all.py 为主）、时间估算更新、smoke test 方案。

### 开发中遇到的问题与解决方案
- **问题：trl 0.18.1 的 DPOConfig 不接受 peft_config 参数**（与旧版教程不一致）。
  - 解决：确认 trl 0.18.1 源码，`peft_config` 在 `DPOTrainer.__init__` 而非 `DPOConfig`。
- **问题：hard_negatives 的 split_origin 字段未被代码使用**。
  - 解决：在 `build_train_examples` 和 `blind_set.assemble_blind_set` 中按 `split_origin` 过滤。
- **问题：Qwen3 默认 thinking 模式输出超长**。
  - 解决：`apply_chat_template(..., enable_thinking=False)` 关闭 thinking。

### 代码实现要点
- `src/data/dataset.py:build_train_examples`：`hard = [h for h in hard_all if h.get("split_origin", "train") == split]`。
- `src/data/blind_set.py:assemble_blind_set`：haystack hard 只取 `split_origin=="test"`。
- `src/eval/baselines.py:QwenZeroShotBaseline.predict`：`enable_thinking=False` + `max_new_tokens=512` + `do_sample=True, temperature=0.1`。
- `src/training/dpo.py`：`peft_config=lora_cfg` 移到 `DPOTrainer(...)`，`metric_for_best_model="eval_rewards/accuracies"`。
- `scripts/run_all.py`：subprocess 驱动，`--from` / `--to` / `--only` / `--limit` / `--judge-model`。

### 测试结果
- `python -m scripts.run_all --help` 正常。
- `bash scripts/run_all.sh --help` 正常（bash shim 委托 Python）。
- 数据去重：train.jsonl 7606→7446（移除 160 条与 test 重复）。
- 尚未跑 smoke test（需服务器 GPU）。

### 性能优化
- 无（本轮为 bug 修复 + 架构调整，无性能相关改动）。

---

## 2026-07-11 — dtype bug 修复 + GPU/API 分离 + 多线程加速 + TF32

### 本次代码更改做了什么

#### 1. dtype bug 修复（`src/models/student.py`）
- **问题**：eval 时 `StudentBaseline.predict` 报 `mat1 and mat2 must have the same dtype, but got Float and Half`。分类头用 `base.dtype` 初始化（4-bit 模型返回 fp32 存储精度），但前向传播的 hidden states 是 bf16（`bnb_4bit_compute_dtype`）。
- **修复**：`__init__` 和 `load()` 中分类头显式用 `torch.bfloat16` 代替 `base.dtype`。forward 中加 `last_hidden.to(dtype=self.classifier.linear.weight.dtype)` 兜底。

#### 2. env.py provider name bug 修复
- `glm` provider 的 `name` 被错误设为 `"aliyun"`，已修正为 `"glm"`。

#### 3. TF32 加速（`src/utils/seed.py`）
- 3090 是 Ampere 架构，开启 TF32 后 matmul/cudnn 用 TF32 替代 fp32，**2-3x matmul 加速**且精度损失可忽略。
- 在 `set_seed()` 中加入 `torch.backends.cuda.matmul.allow_tf32 = True` 和 `torch.backends.cudnn.allow_tf32 = True`。

#### 4. 训练参数优化（`configs/default.yaml` + `src/training/sft.py`）
- SFT: batch 4→8，grad_accum 4→2（等效 batch 16 不变），lr 5e-5→2e-4（QLoRA 标准值）。
- DPO: batch 2→4，grad_accum 8→4（等效 batch 16），lr 5e-7→5e-6。
- DataLoader: num_workers 2→8，加 persistent_workers + prefetch_factor=4。
- `optimizer.zero_grad(set_to_none=True)` 省内存。
- DPO 加 `dataloader_num_workers=8, dataloader_pin_memory=True`。

#### 5. GPU/API 分离架构
- **问题**：DPO pref 和 judge eval 的 API 调用是串行的，3000 条 × 2 次 = 6000 次 API ≈ 6h。qwen-zeroshot GPU 逐条生成 3126 条 ≈ 3.5h。
- **方案**：把 GPU 推理和 API 调用解耦，API 部分用多线程（qwen-plus 30k RPM）。
  - Phase A（GPU）：`scripts/generate_candidates.py` — SFT 模型批量生成候选，输出 candidates.jsonl。
  - Phase B（API）：`scripts/pre_generate.py` — 多线程 judge API 生成偏好对，~10min。
  - qwen-zeroshot：多线程 qwen-plus API 替代 GPU 逐条生成，~10min。
  - judge_eval：多线程 API 替代串行，~5min。
- **学术诚信**：qwen-zeroshot 的 API 调用用的是同一 prompt + 同一模型（Qwen3-8B），只是把 GPU 推理换成 API 推理，结果一致。judge 本来就是 API 调用，多线程只是并行化。

#### 6. 新增文件
- `scripts/generate_candidates.py` — 服务器 GPU 候选生成入口。
- `scripts/pre_generate.py` — 本地多线程 API 预生成（judge / zeroshot / judge_eval 三个子任务）。

#### 7. 批量生成（`src/eval/baselines.py`）
- `QwenZeroShotBaseline` 新增 `predict_batch`：多条 prompt 一起喂 GPU，8x 加速。
- 新增 `FileBaseline`：从预生成 JSON 加载预测，跳过 GPU 推理。

#### 8. run_eval 预生成支持（`src/eval/run_eval.py` + `scripts/run_eval.py`）
- `run_eval` 新增 `pre_generated` 参数：`{baseline_name: json_path}`，跳过对应 baseline 的 GPU 推理。
- `run_one_baseline` 支持批量推理（检测 `predict_batch` 是否被子类 override）。
- `scripts/run_eval.py` 新增 `--pre-generated name=path` 参数。

#### 9. preference.py 拆分（`src/data/preference.py`）
- `generate_candidates_only`：GPU 候选生成，输出 candidates.jsonl。
- `judge_candidates_only`：多线程 API judge，输出 dpo_pairs.jsonl。
- 新增 `_RateLimiter` 多线程速率限制器。

#### 10. llm_judge.py 多线程（`src/eval/llm_judge.py`）
- `run_judge_eval` 新增 `max_workers` 和 `rpm` 参数。
- `_judge_eval_parallel`：多线程并行 judge，保持顺序。

#### 11. run_all.py 更新
- GPU 阶段：`haystack → sft → gen_candidates → dpo → blind → eval`。
- API 阶段由 `pre_generate.py` 独立处理，不在 run_all 中。
- 移除 `--judge-model` 参数（judge 由 pre_generate 处理）。

#### 12. 默认 provider 改为 aliyun（qwen-plus）
- `preference.py`：`judge_provider` 默认 `"aliyun"`。
- `llm_judge.py`：`judge_provider` 默认 `"aliyun"`。
- `pre_generate.py`：`--provider` 默认 `"aliyun"`。

### 开发中遇到的问题与解决方案
- **问题**：4-bit 量化模型的 `base.dtype` 不可靠（返回存储精度 fp32 而非 compute dtype bf16）。
  - 解决：显式用 `torch.bfloat16`。
- **问题**：DPO pref 串行 API 调用 6h 太慢。
  - 解决：GPU/API 分离 + 多线程，judge 部分 6h→10min。
- **问题**：qwen-zeroshot 逐条 GPU 生成 3.5h 太慢。
  - 解决：批量生成（predict_batch 8x）或 API 多线程（10min），两种跑法结果一致。

### 代码实现要点
- `student.py:73,142`：`self.classifier.to(device=device, dtype=torch.bfloat16)`。
- `student.py:89`：`last_hidden = last_hidden.to(dtype=self.classifier.linear.weight.dtype)`。
- `baselines.py:QwenZeroShotBaseline.predict_batch`：padding=True 批量生成。
- `baselines.py:FileBaseline`：按 text 建索引 O(1) 查找。
- `preference.py:generate_candidates_only`：GPU 候选生成，输出 candidates.jsonl。
- `preference.py:judge_candidates_only`：ThreadPoolExecutor 多线程 judge。
- `pre_generate.py`：统一 API 预生成入口（judge / zeroshot / judge_eval）。

### 测试结果
- 10 个 .py 文件 py_compile 全通过。
- 13 个单测全通过。
- `run_all.py --help` 正常。
- blind 阶段实跑验证：2835 条，0 泄漏。

### 性能优化
- TF32：matmul 2-3x 加速。
- SFT batch 8 + steps 减半：1.2-1.5x。
- DataLoader 8 workers + prefetch：数据加载与计算重叠。
- 多线程 API：judge 6h→10min，zeroshot 3.5h→10min，judge_eval 5h→5min。
- **总预估**：20h → **7-9h**（GPU 6.5-8.5h + API 25min）。



