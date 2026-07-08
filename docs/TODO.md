# TODO — 服务器跑训练前的待办清单

> 更新于 2026-07-08，环境重建 + Phase C/D 完成 + 基座换成 Qwen3-8B。

## 当前已完成（本地代码层全部就绪）
- [x] ML 环境重建（pip + 清华源，torch/bitsandbytes CUDA 实测通过）
- [x] 基座模型 Llama-3-8B → Qwen3-8B（Apache 2.0，无申请门槛）
- [x] 合成数据已入仓（`data/synthesized/`，clone 即有，无需重调 API 造数）
- [x] Phase A 数据修复（hard_negatives 展开 + probability patch + WildChat 接入）
- [x] Phase C DPO 接续（候选生成器替换 / cls head 保留 / dpo-only baseline）
- [x] Phase D 评估可视化（混淆矩阵 PNG + CSV + 柱状图 + 延迟表 + judge JSON 修复）
- [x] Dockerfile（`nvidia/cuda:12.4.1` 基础镜像，`docker build -t paper-ml:latest .`）
- [x] 代码已推送到 main 分支

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

### 3. 配置 `.env`（LLM API key，用于 pref 阶段 judge + implicit 阶段）
```bash
cp .env.example .env
# 编辑 .env 填入 GLM_API_KEY / AGNES_API_KEY / GLM_MODEL_NAME 等
```

### 4. 校准配置（服务器 GPU 强，恢复推荐参数）
编辑 `configs/default.yaml`，把 RTX 4060 的保守参数改为服务器推荐值：
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

### 5. 合成数据已入仓（无需重跑）
`data/synthesized/` 下的文件已提交到 git，clone 即有：
- `train.jsonl`（1005 条隐式 Threat 样本，已做 probability patch）
- `test.jsonl`（226 条隐式 Threat 测试样本）
- `hard_negatives.jsonl`（1231 条 Safe 硬负样本，由 `run_hard_negatives` 从上面两个文件展开）

服务器上不需要重新调 LLM API 造数（synth 阶段），直接从 hardneg / pref / sft 开始。

---

## 运行方式一：一键全流程（推荐）

```bash
conda activate ML  # 或直接在 Docker 容器里
bash scripts/run_all.sh all
```

`all` 会按顺序跑：
```
haystack → synth → hardneg → pref → sft → dpo → implicit → blind → eval → judge
```

> **注意**：`synth` 和 `hardneg` 在服务器上不需要重跑（数据已入仓）。`run_all.sh all` 会重跑 synth（覆盖 API），建议**分阶段跑**（见下方），跳过 synth 和 hardneg。

---

## 运行方式二：分阶段跑（推荐，可跳过已有数据的阶段）

每个阶段用 `bash scripts/run_all.sh <stage>` 或直接 `python -m scripts.<module>`。

### 需要 GPU 的阶段
| 阶段 | stage 名 | bash 脚本 | python 命令 | 依赖 | 说明 |
|---|---|---|---|---|---|
| 偏好对生成 | `pref` | `bash scripts/run_all.sh pref` | `python -m scripts.run_preference --judge glm --judge-model glm-4-flash` | SFT checkpoint | 需要 SFT 先跑完；用 LLM judge 生成偏好对 |
| SFT | `sft` | `bash scripts/run_all.sh sft` | `python -m scripts.run_sft` | 合成数据（已入仓） | QLoRA 监督微调，3 epoch |
| DPO | `dpo` | `bash scripts/run_all.sh dpo` | `python -m scripts.run_dpo` | SFT checkpoint + 偏好对 | DPO 对齐 |
| 隐式内化 | `implicit` | `bash scripts/run_all.sh implicit` | `python -m scripts.run_implicit_cot` | SFT checkpoint | Stepwise Internalization，20 epoch，最耗时 |
| 评估 | `eval` | `bash scripts/run_all.sh eval` | `python -m scripts.run_eval` | 盲测集 + 6 个 baseline checkpoint | 生成图表和指标 |
| Judge 评估 | `judge` | `bash scripts/run_all.sh judge` | `python -m scripts.run_judge_eval --predictions outputs/eval/predictions_<name>.json --judge glm` | eval 输出 | LLM-as-judge 质量评估 |

### 不需要 GPU 的阶段（纯本地数据处理）
| 阶段 | stage 名 | bash 脚本 | python 命令 | 依赖 | 说明 |
|---|---|---|---|---|---|
| 草垛下载 | `haystack` | `bash scripts/run_all.sh haystack` | `python -m scripts.prepare_haystack --n 5000` | HF token + WildChat 权限 | 5000 条英文草垛 |
| 硬负样本展开 | `hardneg` | `bash scripts/run_all.sh hardneg` | `python -m scripts.run_hard_negatives` | 合成数据（已入仓） | 已入仓，不需要重跑 |
| 盲测集组装 | `blind` | `bash scripts/run_all.sh blind` | `python -m scripts.run_blind_set` | 草垛 + 硬负样本 | 组装盲测集 |

### 推荐的服务器执行顺序（跳过 synth/hardneg）

```bash
# 1. 草垛（一次性，需要 HF token）
bash scripts/run_all.sh haystack

# 2. SFT 训练（核心，耗 GPU）
bash scripts/run_all.sh sft

# 3. 偏好对生成（依赖 SFT checkpoint + LLM judge API）
bash scripts/run_all.sh pref

# 4. DPO 训练（依赖 SFT checkpoint + 偏好对）
bash scripts/run_all.sh dpo

# 5. 隐式内化（依赖 SFT checkpoint，最耗时）
bash scripts/run_all.sh implicit

# 6. 盲测集组装
bash scripts/run_all.sh blind

# 7. 评估（跑 6 个 baseline + 渲染图表）
bash scripts/run_all.sh eval

# 8. Judge 评估（LLM-as-judge 质量评估）
bash scripts/run_all.sh judge
```

或用 python 命令（等价，更灵活）：
```bash
python -m scripts.prepare_haystack --n 5000
python -m scripts.run_sft
python -m scripts.run_preference --judge glm --judge-model glm-4-flash
python -m scripts.run_dpo
python -m scripts.run_implicit_cot
python -m scripts.run_blind_set
python -m scripts.run_eval
python -m scripts.run_judge_eval --predictions outputs/eval/predictions_implicit-cot.json --judge glm
```

### 用 tmux 后台跑（推荐，防止 SSH 断线）
```bash
tmux new -s train
bash scripts/run_all.sh sft 2>&1 | tee outputs/sft.log
# Ctrl+B D 脱离，docker attach paper-train 重连
```

---

## 预期耗时（RTX 4090 24GB 估算）

| 阶段 | 耗时 | 备注 |
|---|---|---|
| ~~haystack~~ | ~~5-15 min~~ | 已有草垛数据时可跳过（需 HF 下载时才跑） |
| ~~synth~~ | ~~1-3 h~~ | **已入仓，不用重跑** |
| ~~hardneg~~ | ~~< 1 min~~ | **已入仓，不用重跑** |
| pref | 2-6 h | SFT 采样 + LLM judge × 2（位置交换），1005 条 × 2 次 API |
| sft | 4-8 h | QLoRA 3 epoch，2236 样本 |
| dpo | 1-3 h | QLoRA 1 epoch |
| implicit | 12-24 h | Stepwise Internalization 20 epoch，最耗时 |
| blind | < 1 min | 本地组装 |
| eval | 30-60 min | 6 baseline × 1456 条盲测 |
| judge | 1-3 h | 每个 baseline 的 predictions 跑 LLM judge |

**总计约 20-36 小时**（去掉 synth + hardneg）

---

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

1. **pref 阶段 judge 成本**：1005 条 × 2 次位置交换 = 2010 次 judge API 调用，免费 Teacher 质量弱，注意位置偏差。
2. **implicit 阶段显存**：Stepwise Internalization 多轮重训，RTX 4060 8GB 必溢出；服务器 24GB+ 才稳。
3. **Qwen3 权限**：`Qwen/Qwen3-8B` 开源 Apache 2.0，无需申请（已规避 Meta Llama 申请被拒问题）。
4. **数据集 download 缓存**：HF datasets 默认缓存到 `~/.cache/huggingface`，跑完后可清理。
5. **synth 阶段如果要重跑**：数据已入仓，除非想换 Teacher 模型重新造数才需要跑。

## 论文写作待办（代码跑通后）
- [ ] 跑完 eval 取 `outputs/eval/` 全部图表
- [ ] 用 `metrics_table.csv` 填论文 Table 1
- [ ] 用 `tpr_fpr_bars.png` 作论文核心结果图
- [ ] 用 `latency_table.md` 填论文 Table 2（显式 vs 隐式延迟）
- [ ] 用 `confusion_matrix_*.png` 作论文图 1（混淆矩阵对比）
- [ ] 用 `judge_eval_*.json` 的 S1/S2 + bias 数据写 judge 一致性段落
- [ ] 写 Methodology §3.2 probability 字段说明（已改为 Teacher 弱参考，不进损失）
- [ ] 写 Methodology §6.1 baseline 列表（已改为 6 项，移除 roberta-large）
