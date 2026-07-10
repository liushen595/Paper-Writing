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

流水线拆分为两类，GPU 和 API 解耦并行。

### 文件路径一览（关键！按流水线顺序）

```
                         ┌─────── 服务器 3090 ───────┐    ┌─── 拷贝到本地 ───┐    ┌─── 本地 API ───┐    ┌── 拷回服务器 ──┐
SFT 训练                  checkpoints/sft/best/        │                       │                   │                  │
                         │ adapter_model.safetensors   │                       │                   │                  │
                         │ classifier_head.pt          │                       │                   │                  │
gen_candidates ← SFT 读  data/preference/candidates.jsonl ──scp──▶ 本地同路径   │                   │                  │
                         │                            │                       │                   │                  │
judge (API)              │                            │                       │ 本地读取 candidates │ dpo_pairs.jsonl  │
                         │                            │                       │                    │ ──scp──▶ 服务器   │
DPO 训练 ← SFT + pairs   checkpoints/dpo/             │                       │                   │                  │
                         │ adapter_model.safetensors   │                       │                   │                  │
blind                    data/blind/test_blind.csv     │                       │                   │                  │
eval                     outputs/eval/                 │                       │                   │                  │
                         │ predictions_*.json          │                       │                   │                  │
                         │ metrics_table.md            │                       │                   │                  │
                         │ *.png                       │                       │                   │                  │
                         │                            │                       │                   │                  │
judge_eval (API)         │                            │ predictions_*.json    │ judge_eval_*.json │                  │
                         │                            │ ──scp──▶ 本地          │                   │ (最终产物，论文用) │
```

### 阶段说明

| # | 阶段 | 位置 | 输入 | 输出 |
|---|---|---|---|---|
| 1 | haystack | **服务器** | 网络下载 | `data/haystack/wildchat_nontoxic.jsonl` |
| 2 | sft | **服务器** | `data/synthesized/train.jsonl` | `checkpoints/sft/` |
| 3 | gen_candidates | **服务器** | `checkpoints/sft/best/` | `data/preference/candidates.jsonl` |
| 4 | **拷贝** | — | — | `scp server:.../candidates.jsonl ./data/preference/` |
| 5 | judge | **本地** | `data/preference/candidates.jsonl` | `data/preference/dpo_pairs.jsonl` |
| 6 | **拷贝** | — | — | `scp ./data/preference/dpo_pairs.jsonl server:.../` |
| 7 | dpo | **服务器** | `checkpoints/sft/` + `dpo_pairs.jsonl` | `checkpoints/dpo/` |
| 8 | blind | **服务器** | `hard_negatives.jsonl` + haystack | `data/blind/test_blind.csv` |
| 9 | eval | **服务器** | `test_blind.csv` + 各 checkpoint | `outputs/eval/predictions_*.json` 等 |
| 10 | **拷贝** | — | — | `scp server:.../predictions_*.json ./outputs/eval/` |
| 11 | judge_eval | **本地** | `outputs/eval/predictions_*.json` | `outputs/eval/judge_eval_*.json` |

### 精确拷贝命令（按需替换路径）

```bash
# === 服务器 3090 跑完 gen_candidates 后 ===
# 从服务器拷贝到本地：
scp user@server:/home/user/Paper-Writing/data/preference/candidates.jsonl ./data/preference/

# === 本地跑完 judge 后 ===
# 从本地拷贝回服务器：
scp ./data/preference/dpo_pairs.jsonl user@server:/home/user/Paper-Writing/data/preference/

# === 服务器跑完 eval 后 ===
# 从服务器拷贝 predictions 到本地（做 judge_eval）：
scp user@server:/home/user/Paper-Writing/outputs/eval/predictions_*.json ./outputs/eval/
```

### 完整执行命令

```bash
# ========== 服务器 3090（GPU 阶段）==========

# 1. 草垛下载（一次性）
python -m scripts.run_all --only haystack

# 2. SFT 训练 → checkpoints/sft/
python -m scripts.run_all --only sft

# 3. DPO 候选生成（GPU 推理） → data/preference/candidates.jsonl
python -m scripts.run_all --only gen_candidates --limit 3000

# ═══════ 拷贝 candidates.jsonl 到本地 ═══════
# scp server:.../data/preference/candidates.jsonl ./data/preference/

# ========== 本地（API 阶段，多线程）==========

# 4. 多线程 judge → data/preference/dpo_pairs.jsonl
python -m scripts.pre_generate judge --input data/preference/candidates.jsonl --max-workers 10

# ═══════ 拷贝 dpo_pairs.jsonl 回服务器 ═══════
# scp ./data/preference/dpo_pairs.jsonl server:.../data/preference/

# ========== 服务器 3090（GPU 阶段）==========

# 5. DPO 训练 → checkpoints/dpo/
python -m scripts.run_all --only dpo

# 6. 盲测集 → data/blind/test_blind.csv
python -m scripts.run_all --only blind

# 7. 评估 → outputs/eval/predictions_*.json, metrics_table.md
python -m scripts.run_all --only eval

# ═══════ 拷贝 predictions 到本地 ═══════
# scp server:.../outputs/eval/predictions_*.json ./outputs/eval/

# ========== 本地（API 阶段，多线程）==========

# 8. Judge 质量评估 → outputs/eval/judge_eval_predictions_*.json
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_sft-no-dpo.json --max-workers 10
python -m scripts.pre_generate judge_eval --input outputs/eval/predictions_dpo-only.json --max-workers 10
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

1. **gen_candidates 是 GPU 瓶颈**：SFT 模型批量生成 2 候选，3000 条 ≈ 1-1.5h（batch_size=16）。
2. **Phase 3 隐式内化**：本轮未跑，代码已实现（`src/training/implicit_cot.py`），列为 future work。
3. **Qwen3 权限**：开源 Apache 2.0，无需申请。

## 论文写作待办（代码跑通后）
- [ ] 跑完 eval 取 `outputs/eval/` 全部图表
- [ ] 用 `metrics_table.csv` 填论文 Table 1
- [ ] 用 `tpr_fpr_bars.png` 作论文核心结果图
- [ ] 用 `latency_table.md` 填论文 Table 2
- [ ] 用 `confusion_matrix_*.png` 作论文图 1
- [ ] 用 `judge_eval_*.json` 的 S1/S2 + bias 数据写 judge 一致性段落
- [ ] 写 Methodology §5.3 Phase 3 future work 段落
- [ ] 写 Methodology §6.1 baseline 列表（3 项：toxic-bert / sft-no-dpo / dpo-only）
