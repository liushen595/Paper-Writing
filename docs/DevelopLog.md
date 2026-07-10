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


