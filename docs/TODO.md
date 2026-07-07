# TODO — 服务器跑训练前的待办清单

> 写于 2026-07-07，环境重建 + Phase C/D 完成后的状态快照。

## 当前已完成（本地代码层全部就绪）
- [x] ML 环境重建（pip + 清华源，torch/bitsandbytes CUDA 实测通过）
- [x] Phase A 数据修复（hard_negatives 展开 + probability patch + WildChat 接入）
- [x] Phase C DPO 接续（候选生成器替换 / cls head 保留 / dpo-only baseline）
- [x] Phase D 评估可视化（混淆矩阵 PNG + CSV + 柱状图 + 延迟表 + judge JSON 修复）
- [x] pytest 13/13 通过
- [x] environment.yml 重写为 pip 段格式
- [x] 代码已推送到 main 分支（commit f447d14）

## 服务器跑训练前必做（人工）

### 1. 服务器环境准备
```bash
# clone 仓库
git clone https://github.com/liushen595/Paper-Writing.git
cd Paper-Writing

# 按 environment.yml 重建环境（注意：必须用 pip，不要用 conda install 装包）
conda create -n ML python=3.10 -y
conda activate ML
# 一行装完（清华源，国内服务器加速；海外服务器去掉 -i 参数）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "torch==2.5.1" "torchvision==0.20.1" \
  "transformers==4.44.1" "datasets==4.4.1" "peft==0.19.1" "trl==0.12.2" \
  "accelerate==1.14.0" "bitsandbytes==0.49.2" "tokenizers==0.19.1" \
  "huggingface-hub==0.34.4" "safetensors==0.7.0" "sentencepiece==0.2.0" \
  "numpy==2.0.1" "pandas==2.3.3" "scipy==1.15.3" "scikit-learn==1.7.2" \
  "matplotlib==3.10.6" "seaborn==0.13.2" "pyyaml==6.0.3" \
  "requests==2.32.5" "openai==2.30.0" "python-dotenv==1.2.1" \
  "tqdm==4.67.1" "pydantic==2.13.4" "rich==14.2.0" \
  "jieba==0.42.1" "nltk==3.9.2" "timm==1.0.22" \
  "rouge-score==0.1.2" "evaluate==0.4.6" "statsmodels==0.14.6" \
  "pytest==9.0.3"
```

### 2. HuggingFace 登录（必需，WildChat 是 gated dataset）
```bash
conda run -n ML huggingface-cli login
# 粘贴 HF token（在 https://huggingface.co/settings/tokens 创建，read 权限即可）
```
并在浏览器申请以下数据集权限（通常几小时内通过）：
- https://huggingface.co/datasets/allenai/WildChat-nontoxic （草垛）
- https://huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct （基座模型，Llama 系列通常需申请）

### 3. 配置 `.env`（LLM API key，用于造数 + judge）
```bash
cp .env.example .env
# 编辑 .env 填入 GLM_API_KEY / AGNES_API_KEY / GLM_MODEL_NAME 等
```

### 4. 校准配置（服务器 GPU 强，恢复推荐参数）
编辑 `configs/default.yaml`：
- `sft.per_device_batch_size`: 1 → **4**（RTX 4090 24GB）/ **8**（A6000 48GB）
- `sft.gradient_accumulation_steps`: 16 → **4**
- `sft.max_seq_len`: 512 → **1024**
- `sft.lora_r`: 16 → **64**
- `dpo.per_device_batch_size`: 1 → **2** / **4**
- `dpo.gradient_accumulation_steps`: 16 → **8**
- `dpo.max_length`: 512 → **1024**
- `implicit_cot.per_device_batch_size`: 1 → **4**
- `implicit_cot.gradient_accumulation_steps`: 16 → **4**
- `implicit_cot.max_seq_len`: 512 → **1024**

## 服务器一键跑全流程

```bash
conda activate ML
bash scripts/run_all.sh all
```

`all` 会按顺序跑：`haystack → synth → hardneg → pref → sft → dpo → implicit → blind → eval → judge`

### 关于草垛数据会不会自己造？
**会自动跑**。`run_all.sh all` 第一步就是 `run_haystack`，会调 `scripts/prepare_haystack.py`：
1. 从 HuggingFace 下载 `allenai/WildChat-nontoxic`（约 200-400MB）
2. 过滤 English + 非 redacted，取第一条 user message
3. seed=42 随机采样 5000 条写到 `data/haystack/wildchat_nontoxic.jsonl`
4. 后续 `blind` 阶段会自动从该文件读取草垛混入盲测集

**前提**：第 2 步已完成 `huggingface-cli login` 且申请了 WildChat-nontoxic 权限。否则会卡在第一步报 401。

### 草垛单独跑（验证）
```bash
conda run -n ML python -m scripts.prepare_haystack --n 5000
# 验证
wc -l data/haystack/wildchat_nontoxic.jsonl  # 应为 5000
```

### 单阶段跑（推荐先小规模冒烟）
```bash
# 先跑数据阶段（不耗 GPU，验证数据流通）
bash scripts/run_all.sh haystack
bash scripts/run_all.sh hardneg
bash scripts/run_all.sh blind

# 再跑训练（耗 GPU）
bash scripts/run_all.sh sft
bash scripts/run_all.sh pref   # 依赖 sft checkpoint
bash scripts/run_all.sh dpo    # 依赖 sft checkpoint
bash scripts/run_all.sh implicit  # 依赖 sft checkpoint, 工时长

# 最后评估
bash scripts/run_all.sh eval
bash scripts/run_all.sh judge
```

## 预期耗时（RTX 4090 24GB 估算）
| 阶段 | 耗时 | 备注 |
|---|---|---|
| haystack | 5-15 min | WildChat 下载 + 过滤 |
| synth | 1-3 h | 1005 条 DOJ → LLM 造数，受 API 速率限制 |
| hardneg | < 1 min | 纯本地数据展开 |
| pref | 2-6 h | SFT 采样 + LLM judge × 2（位置交换），1005 条 × 2 次 API |
| sft | 4-8 h | QLoRA 3 epoch，2236 样本 |
| dpo | 1-3 h | QLoRA 1 epoch |
| implicit | 12-24 h | Stepwise Internalization 20 epoch，最耗时 |
| blind | < 1 min | 本地组装 |
| eval | 30-60 min | 6 baseline × 1456 条盲测 |
| judge | 1-3 h | 每个 baseline 的 predictions 跑 LLM judge |

**总计约 24-40 小时**，建议用 `nohup` 或 `tmux` 后台跑：
```bash
tmux new -s train
bash scripts/run_all.sh all 2>&1 | tee outputs/run_all.log
# Ctrl+B D 脱离
```

## 训练完成后取论文素材

`outputs/eval/` 目录下直接拿：
- `metrics_table.md` — 指标矩阵（论文 Table 1 来源）
- `metrics_table.csv` — Table 1 机器可读版
- `tpr_fpr_bars.png` — TPR/FPR 跨 baseline 对比图（论文核心结果图）
- `confusion_matrix_<baseline>.png` — 每个 baseline 一张混淆矩阵（论文图 1 系列）
- `latency_table.md` — 显式 vs 隐式延迟对比 + 加速比（论文 Table 2）
- `predictions_<baseline>.json` — 原始预测，供分析
- `judge_eval_predictions_<baseline>.json` — LLM-as-judge 质量 + 偏差监控

## 已知风险 / 注意事项

1. **synth 阶段 API 速率**：1005 条 DOJ 造数受 LLM API 速率限制，若用 GLM 免费档可能 3-5 小时。可加 `--limit 50` 先冒烟。
2. **pref 阶段 judge 成本**：1005 条 × 2 次位置交换 = 2010 次 judge API 调用，免费 Teacher 质量弱，注意位置偏差。
3. **implicit 阶段显存**：Stepwise Internalization 多轮重训，RTX 4060 8GB 必溢出；服务器 24GB+ 才稳。
4. **Llama-3 权限**：`meta-llama/Meta-Llama-3-8B-Instruct` 也是 gated，记得申请。
5. **数据集 download 缓存**：HF datasets 默认缓存到 `~/.cache/huggingface`，跑完后可清理（`rm -rf ~/.cache/huggingface/datasets/allenai_wildchat*`）。

## 论文写作待办（代码跑通后）
- [ ] 跑完 eval 取 `outputs/eval/` 全部图表
- [ ] 用 `metrics_table.csv` 填论文 Table 1
- [ ] 用 `tpr_fpr_bars.png` 作论文核心结果图
- [ ] 用 `latency_table.md` 填论文 Table 2（显式 vs 隐式延迟）
- [ ] 用 `confusion_matrix_*.png` 作论文图 1（混淆矩阵对比）
- [ ] 用 `judge_eval_*.json` 的 S1/S2 + bias 数据写 judge 一致性段落
- [ ] 写 Methodology §3.2 probability 字段说明（已改为 Teacher 弱参考，不进损失）
- [ ] 写 Methodology §6.1 baseline 列表（已改为 6 项，移除 roberta-large）
