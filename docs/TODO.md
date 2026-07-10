# TODO — 服务器跑训练前的待办清单

> 更新于 2026-07-11，Phase 3 隐式 CoT 内化退役为 future work，流水线精简为 4 baseline。

## 当前已完成（本地代码层全部就绪）
- [x] ML 环境重建（pip + 清华源，torch/bitsandbytes CUDA 实测通过）
- [x] 基座模型 Llama-3-8B → Qwen3-8B（Apache 2.0，无申请门槛）
- [x] 合成数据已入仓（`data/synthesized/`，clone 即有，无需重调 API 造数）
- [x] Phase A 数据修复（hard_negatives 展开 + probability patch + WildChat 接入）
- [x] Phase C DPO 接续（候选生成器替换 / cls head 保留 / dpo-only baseline）
- [x] Phase D 评估可视化（混淆矩阵 PNG + CSV + 柱状图 + 延迟表 + judge JSON 修复）
- [x] Dockerfile（`nvidia/cuda:12.4.1` 基础镜像，`docker build -t paper-ml:latest .`）
- [x] 代码已推送到 main 分支
- [x] **数据泄漏修复**：hard_negatives 按 `split_origin` 字段分离 train/test；train.jsonl 去重（移除与 test 重复的 160 条）
- [x] **DPO bug 修复**：peft_config 移至 DPOTrainer；metric_for_best_model 改 `eval_rewards/accuracies`
- [x] **Qwen baseline 修复**：关闭 thinking 模式（`enable_thinking=False`）；max_new_tokens 128→512；token 计数修正
- [x] **run_all 重写**：Python 驱动器 `scripts/run_all.py`，支持 `--from` / `--only` / `--to` / `--limit`
- [x] **--limit 参数**：sft / pref / eval / judge 均支持，用于 smoke test
- [x] **Phase 3 退役**：implicit-cot baseline 移除，改为 future work

## 服务器跑训练前必做（人工）

### 1. 服务器环境准备（**推荐用 Docker，见 [docs/Docker.md](Docker.md)**）

#### 方案 A：Docker（推荐，完美复刻本地环境）
```bash
# 本地构建并推送镜像（一次性，见 Docker.md）
docker build -t paper-ml:latest .
docker push <your-registry>/paper-ml:latest

# 服务器拉取镜像 + clone 代码
docker pull <your-registry>/paper-ml:latest
git clone https://github.com/liushen595/Paper-Writing.git && cd Paper-Writing
docker run --gpus all -it --rm --shm-size=16g \
  -v $(pwd):/workspace -w /workspace \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.config/huggingface:/root/.config/huggingface \
  paper-ml:latest bash
```

#### 方案 B：手动 conda + pip（无 Docker 时）
```bash
git clone https://github.com/liushen595/Paper-Writing.git && cd Paper-Writing
conda create -n ML python=3.10 -y && conda activate ML
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
python scripts/check_imports.py  # 验证：98 通过 + 12 跳过 = 0 失败
```

### 2. HuggingFace 登录（WildChat 草垛必需）
```bash
conda run -n ML huggingface-cli login  # 粘贴 HF token
```
并在浏览器申请：https://huggingface.co/datasets/allenai/WildChat-nontoxic

> 基座模型 `Qwen/Qwen3-8B` 开源 Apache 2.0，无需申请。

### 3. 配置 `.env`（LLM API key，用于 pref 阶段 judge + judge 评估）
```bash
cp .env.example .env
# 编辑 .env 填入 GLM_API_KEY / AGNES_API_KEY / GLM_MODEL_NAME 等
```

### 4. 校准配置（服务器 GPU 强，恢复推荐参数）
编辑 `configs/default.yaml`，把 RTX 4060 的保守参数改为服务器推荐值：
- `sft.per_device_batch_size`: 1 → **4**（RTX 3090 24GB）
- `sft.gradient_accumulation_steps`: 16 → **4**
- `sft.max_seq_len`: 512 → **1024**
- `sft.lora_r`: 16 → **64**
- `dpo.per_device_batch_size`: 1 → **2**
- `dpo.gradient_accumulation_steps`: 16 → **8**
- `dpo.max_length`: 512 → **1024**

### 5. 合成数据已入仓（无需重跑）
`data/synthesized/` 下的文件已提交到 git，clone 即有：
- `train.jsonl`（7446 条隐式 Threat 样本，已去重去泄漏）
- `test.jsonl`（1935 条隐式 Threat 测试样本）
- `hard_negatives.jsonl`（4622 条 Safe 硬负样本，含 `split_origin` 字段区分 train/test）

服务器上不需要重新调 LLM API 造数（synth 阶段），直接从 sft 开始。

---

## 运行方式一：一键全流程（推荐）

```bash
conda activate ML  # 或直接在 Docker 容器里
python -m scripts.run_all
# 或等价的 bash 入口：
bash scripts/run_all.sh
```

流水线按顺序执行：
```
haystack → sft → pref → dpo → blind → eval → judge
```

### 灵活控制
```bash
python -m scripts.run_all --from sft              # 从 sft 开始跑到结尾
python -m scripts.run_all --only sft             # 只跑 sft
python -m scripts.run_all --only pref dpo        # 只跑 pref + dpo
python -m scripts.run_all --from sft --to eval   # sft 到 eval（含两端）
python -m scripts.run_all --limit 200            # 限制样本数（smoke test）
python -m scripts.run_all --judge-model glm-4-flash  # 覆盖 judge 模型
```

---

## 运行方式二：分阶段跑

每个阶段用 `python -m scripts.<module>` 直接跑：

### 需要 GPU 的阶段
| 阶段 | 命令 | 依赖 | 说明 |
|---|---|---|---|
| SFT | `python -m scripts.run_sft` | 合成数据（已入仓） | QLoRA 监督微调，3 epoch |
| 偏好对生成 | `python -m scripts.run_preference --judge glm --judge-model glm-4-flash --limit 3000` | SFT checkpoint | LLM judge 生成偏好对（限制 3000 条） |
| DPO | `python -m scripts.run_dpo` | SFT checkpoint + 偏好对 | DPO 对齐 |
| 评估 | `python -m scripts.run_eval` | 盲测集 + 4 个 baseline checkpoint | 生成图表和指标 |
| Judge 评估 | `python -m scripts.run_judge_eval --predictions outputs/eval/predictions_<name>.json --judge glm --limit 200` | eval 输出 | LLM-as-judge 质量评估（每 baseline 限 200 条） |

### 不需要 GPU 的阶段
| 阶段 | 命令 | 依赖 | 说明 |
|---|---|---|---|
| 草垛下载 | `python -m scripts.prepare_haystack --n 5000` | HF token + WildChat 权限 | 5000 条英文草垛，盲测集用 2000 |
| 盲测集组装 | `python -m scripts.run_blind_set` | 草垛 + 硬负样本 | 组装盲测集（226 needles + 900 test-hard + 2000 WildChat = 3126 条） |

### 推荐的服务器执行顺序（跳过 synth/hardneg）

```bash
# 1. 草垛（一次性，需要 HF token）
python -m scripts.run_all --only haystack

# 2. SFT 训练（核心，耗 GPU）
python -m scripts.run_all --only sft

# 3. 偏好对生成（依赖 SFT checkpoint + LLM judge API）
python -m scripts.run_all --only pref

# 4. DPO 训练（依赖 SFT checkpoint + 偏好对）
python -m scripts.run_all --only dpo

# 5. 盲测集组装
python -m scripts.run_all --only blind

# 6. 评估（跑 4 个 baseline + 渲染图表）
python -m scripts.run_all --only eval

# 7. Judge 评估（LLM-as-judge 质量评估）
python -m scripts.run_all --only judge
```

或一步到位：
```bash
python -m scripts.run_all --from sft
```

### Smoke Test（跑通全流程排 bug，~1h）
```bash
python -m scripts.run_all --from sft --limit 200
# 注：judge 阶段 limit 200 传给每个 baseline 的 predictions
```

### 用 tmux 后台跑（推荐，防止 SSH 断线）
```bash
tmux new -s train
python -m scripts.run_all --from sft 2>&1 | tee outputs/pipeline.log
# Ctrl+B D 脱离，tmux attach -t train 重连
```

---

## 预期耗时（单张 RTX 3090 24GB 估算）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| haystack | 5-15 min | 需 HF 下载 WildChat |
| sft | ~4.2 h | QLoRA 3 epoch，11168 样本（7446 train + 3722 train-hard） |
| pref (limit 3000) | ~7.5 h | SFT 采样 + LLM judge × 2（位置交换），3000 条 × 2 次 API |
| dpo | ~1.5 h | QLoRA 1 epoch，~1500 偏好对 |
| blind | < 1 min | 本地组装 |
| eval | ~4.9 h | 4 baseline × 3126 条盲测 |
| judge (limit 200) | ~1 h | 每个 baseline 的 200 条 predictions 跑 LLM judge |
| **总计** | **~19.5 h** | 37h 内从容完成，留 ~17h 论文写作 |

---

## 训练完成后取论文素材

`outputs/eval/` 目录下直接拿：
- `metrics_table.md` — 指标矩阵（论文 Table 1 来源）
- `metrics_table.csv` — Table 1 机器可读版
- `tpr_fpr_bars.png` — TPR/FPR 跨 baseline 对比图（论文核心结果图）
- `confusion_matrix_<baseline>.png` — 每个 baseline 一张混淆矩阵（论文图 1 系列）
- `latency_table.md` — 延迟对比（论文 Table 2）
- `predictions_<baseline>.json` — 原始预测，供分析
- `judge_eval_predictions_<baseline>.json` — LLM-as-judge 质量 + 偏差监控

## 已知风险 / 注意事项

1. **pref 阶段 judge 成本**：3000 条 × 2 次位置交换 = 6000 次 judge API 调用，免费 Teacher 质量弱，注意位置偏差（已实现一致性过滤）。
2. **Qwen3 权限**：`Qwen/Qwen3-8B` 开源 Apache 2.0，无需申请。
3. **数据集 download 缓存**：HF datasets 默认缓存到 `~/.cache/huggingface`，跑完后可清理。
4. **Phase 3 隐式内化**：本轮未跑，代码已实现（`src/training/implicit_cot.py`），列为 future work。

## 论文写作待办（代码跑通后）
- [ ] 跑完 eval 取 `outputs/eval/` 全部图表
- [ ] 用 `metrics_table.csv` 填论文 Table 1
- [ ] 用 `tpr_fpr_bars.png` 作论文核心结果图
- [ ] 用 `latency_table.md` 填论文 Table 2（各 baseline 延迟对比）
- [ ] 用 `confusion_matrix_*.png` 作论文图 1（混淆矩阵对比）
- [ ] 用 `judge_eval_*.json` 的 S1/S2 + bias 数据写 judge 一致性段落
- [ ] 写 Methodology §3.2 probability 字段说明（已改为 Teacher 弱参考，不进损失）
- [ ] 写 Methodology §6.1 baseline 列表（已改为 4 项，移除 roberta-large + implicit-cot）
- [ ] 写 Methodology §5.3 Phase 3 future work 段落（已退役，说明 delta=8 不足以内化）
