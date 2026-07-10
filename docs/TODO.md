# TODO — 服务器跑训练前的待办清单

> 更新于 2026-07-11，GPU/API 分离架构 + 多线程加速 + Phase 3 退役。

## 当前已完成（本地代码层全部就绪）
- [x] ML 环境重建（pip + 清华源，torch/bitsandbytes CUDA 实测通过）
- [x] 基座模型 Qwen3-8B（Apache 2.0，无申请门槛）
- [x] 合成数据已入仓（`data/synthesized/`，clone 即有）
- [x] 数据泄漏修复（split_origin 过滤 + train 去重）
- [x] DPO bug 修复（peft_config + metric 名 + zero_grad）
- [x] Qwen baseline 修复（thinking 模式 + token 计数 + dtype 修复）
- [x] TF32 加速 + batch 优化（SFT batch=8, DPO batch=4）
- [x] **GPU/API 分离架构**：候选生成在服务器 GPU，judge 调用本地多线程 API
- [x] **多线程 API 工具**：`scripts/pre_generate.py`（judge / zeroshot / judge_eval）
- [x] **批量生成**：QwenZeroShotBaseline 支持 predict_batch（8x 加速）
- [x] **预生成加载**：eval 支持 `--pre-generated` 跳过 GPU baseline
- [x] **run_all 重写**：`scripts/run_all.py`（--from/--to/--only/--limit）
- [x] Phase 3 隐式内化退役为 future work，baseline 精简为 4 项
- [x] 默认 LLM provider 改为 aliyun（qwen-plus），env.py glm 名称修复

## 服务器跑训练前必做（人工）

### 1. 服务器环境准备

#### 方案 A：Docker（推荐）
```bash
docker build -t paper-ml:latest .
docker push <your-registry>/paper-ml:latest
docker pull <your-registry>/paper-ml:latest
git clone https://github.com/liushen595/Paper-Writing.git && cd Paper-Writing
docker run --gpus all -it --rm --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.config/huggingface:/root/.config/huggingface \
  paper-ml:latest bash
```

#### 方案 B：手动 conda + pip
```bash
git clone https://github.com/liushen595/Paper-Writing.git && cd Paper-Writing
conda create -n ML python=3.10 -y && conda activate ML
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python scripts/check_imports.py
```

### 2. HuggingFace 登录（WildChat 草垛必需）
```bash
conda run -n ML huggingface-cli login
```

### 3. 配置 `.env`（LLM API key，本地多线程 judge 用）
```bash
cp .env.example .env
# 编辑 .env，至少填入 DASHSCOPE_API_KEY（阿里云 qwen-plus）
# DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# ALIYUN_MODEL_NAME=qwen-plus
```

### 4. 配置已校准（RTX 3090 级别，无需改）
`configs/default.yaml` 已设为 3090 参数：
- SFT: batch=8, grad_accum=2, lr=2e-4, max_seq_len=1024, lora_r=64
- DPO: batch=4, grad_accum=4, lr=5e-6, max_length=1024

### 5. 合成数据已入仓（无需重跑）
- `train.jsonl`（7446 条，已去重去泄漏）
- `test.jsonl`（1935 条）
- `hard_negatives.jsonl`（4622 条，含 split_origin 字段）

---

## 架构：GPU 阶段（服务器 3090） + API 阶段（本地多线程）

流水线拆分为两类，GPU 和 API 解耦并行：

### GPU 阶段（服务器 3090，用 run_all 驱动）
```
haystack → sft → gen_candidates → dpo → blind → eval
```

### API 阶段（本地/任意机器，多线程，无需 GPU）
```
judge      — 读取 candidates.jsonl，多线程 judge API → dpo_pairs.jsonl
judge_eval — 读取 predictions JSON，多线程 quality judge → judge_eval_*.json
```

> **qwen-zeroshot 不在 API 阶段**——它必须用 Qwen3-8B 模型做 GPU 推理。用 qwen-plus API 替代是学术造假（不同模型/规模）。qwen-zeroshot 由 eval 阶段在 GPU 上批量完成（predict_batch 8x）。

### 完整执行顺序

```bash
# ========== 服务器 3090（GPU 阶段）==========

# 1. 草垛下载（一次性）
python -m scripts.run_all --only haystack

# 2. SFT 训练
python -m scripts.run_all --only sft

# 3. DPO 候选生成（GPU，用 SFT 模型，不调 API）
python -m scripts.run_all --only gen_candidates --limit 3000
# 输出: data/preference/candidates.jsonl

# --- 把 candidates.jsonl 拷到本地 ---

# ========== 本地（API 阶段，多线程）==========

# 4. DPO 偏好对生成（多线程 judge API，~10min）
python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10
# 输出: data/preference/dpo_pairs.jsonl

# --- 把 dpo_pairs.jsonl 拷回服务器 ---

# ========== 服务器 3090（GPU 阶段）==========

# 5. DPO 训练
python -m scripts.run_all --only dpo

# 6. 盲测集组装
python -m scripts.run_all --only blind

# 7. 评估（GPU baseline: toxic-bert + sft-no-dpo + dpo-only）
python -m scripts.run_all --only eval

# --- 把 outputs/eval/predictions_*.json 拷到本地 ---

# ========== 本地（API 阶段，多线程）==========

# 8. qwen-zeroshot 已在步骤 7 的 eval 中自动完成（GPU 批量生成，predict_batch 8x 加速）
# eval 会跑全部 4 个 baseline: toxic-bert + qwen-zeroshot + sft-no-dpo + dpo-only
# qwen-zeroshot 使用 Qwen3-8B 模型 GPU 推理（不许用 API 替代，学术造假）

# 9. Judge 质量评估（多线程 API，~10min/baseline）
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_sft-no-dpo.json --max-workers 10
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_dpo-only.json --max-workers 10
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_qwen-zeroshot.json --max-workers 10
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_toxic-bert.json --max-workers 10
```

### Smoke Test（跑通全流程排 bug，~30min）
```bash
# 服务器：SFT + 候选生成 + DPO + blind + eval（各 limit 200）
python -m scripts.run_all --from sft --to eval --limit 200

# 本地：judge + judge_eval（各 limit 200）
python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --limit 200 --max-workers 10
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_sft-no-dpo.json --limit 200 --max-workers 10
```

### 用 tmux 后台跑（推荐，防止 SSH 断线）
```bash
tmux new -s train
python -m scripts.run_all --from sft --to eval 2>&1 | tee outputs/pipeline.log
```

---

## 预期耗时（单张 RTX 3090 24GB + 本地多线程 API）

### GPU 阶段（服务器 3090）
| 阶段 | 耗时 | 说明 |
|---|---|---|
| haystack | 5-15 min | HF 下载 WildChat |
| sft | ~2-3 h | TF32+batch=8，3 epoch，11168 样本 |
| gen_candidates (3000) | ~2.5 h | SFT 模型生成 2 候选/样本 |
| dpo | ~0.5-1 h | batch=4，~1500 偏好对 |
| blind | < 1 min | 本地组装 |
| eval (GPU baseline) | ~1.5-2 h | toxic-bert + sft-no-dpo + dpo-only，3126 条 |
| **GPU 小计** | **~6.5-8.5 h** | |

### API 阶段（本地多线程，qwen-plus 30k RPM）
| 阶段 | 耗时 | 说明 |
|---|---|---|
| judge (6000 API) | ~10 min | 10 线程，3000 样本 × 2 位置交换 |
| judge_eval (4×200 API) | ~5 min | 10 线程，每 baseline 200 条 |
| **API 小计** | **~15 min** | |

### 总计
| | 服务器 GPU | 本地 API | 总时间 |
|---|---|---|---|
| 实际耗时 | ~6.5-8.5h | ~15min | **~7-9h** |
| 37h 内剩余 | — | — | **~28h 论文写作 + buffer** |

---

## 训练完成后取论文素材

`outputs/eval/` 目录下直接拿：
- `metrics_table.md` / `.csv` — 指标矩阵（Table 1）
- `tpr_fpr_bars.png` — TPR/FPR 跨 baseline 对比图（核心结果图）
- `confusion_matrix_<baseline>.png` — 每个 baseline 一张混淆矩阵（图 1）
- `latency_table.md` — 延迟对比（Table 2）
- `predictions_<baseline>.json` — 原始预测
- `judge_eval_predictions_<baseline>.json` — LLM-as-judge 质量 + 偏差监控

## 已知风险 / 注意事项

1. **gen_candidates 是 GPU 瓶颈**：SFT 模型逐样本生成 2 候选，3000 条 ≈ 2.5h。若太慢可减少 limit 或在本地 4060 上跑（4-bit 推理可容纳 8B）。
2. **qwen-zeroshot 两种跑法**：(a) 服务器 GPU 批量生成（predict_batch 8x 加速，~30-60min）；(b) 本地 API 多线程（~10min）。论文中报告的是同模型 Qwen3-8B 的结果，两种跑法结果一致。
3. **Phase 3 隐式内化**：本轮未跑，代码已实现（`src/training/implicit_cot.py`），列为 future work。
4. **Qwen3 权限**：开源 Apache 2.0，无需申请。

## 论文写作待办（代码跑通后）
- [ ] 跑完 eval 取 `outputs/eval/` 全部图表
- [ ] 用 `metrics_table.csv` 填论文 Table 1
- [ ] 用 `tpr_fpr_bars.png` 作论文核心结果图
- [ ] 用 `latency_table.md` 填论文 Table 2
- [ ] 用 `confusion_matrix_*.png` 作论文图 1
- [ ] 用 `judge_eval_*.json` 的 S1/S2 + bias 数据写 judge 一致性段落
- [ ] 写 Methodology §5.3 Phase 3 future work 段落
- [ ] 写 Methodology §6.1 baseline 列表（4 项）
