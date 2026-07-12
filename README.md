# ThreatWeaver

ThreatWeaver 是一个课程研究项目，用于分析互联网文本中的隐式犯罪意图。系统以 DOJ 新闻稿为语义种子，通过 Teacher LLM 合成间接意图、Safe 对照与 rationale，训练 Qwen3-8B QLoRA 学生模型，并使用自动偏好对执行 DPO。

当前实验是诊断性负结果：SFT 显示有限任务信号，但 DPO 后的 pooled classifier 接近恒定预测 Threat。代码已知存在 SFT 分类头训练/推理输入错位，以及 DPO 更新 LoRA 但不联合更新分类头的问题。本项目不得用于自动执法、惩罚、账号封禁或个人风险评分。

## 归档内容

代码归档包含：

- Python 源码、脚本、测试与集中配置；
- `data/raw/doj_raw.jsonl`：9,582 条 DOJ 原始爬取记录；
- `data/synthesized/`：本次课程实验使用的合成 train/test/hard-negative 数据；
- `artifacts/provenance.json`：公开工件的 SHA-256、大小、记录数和版本缺口；
- `requirements.txt`、`Dockerfile` 和 `.env.example`。

代码归档不包含论文源、开发文档、本地 agent 配置、模型 checkpoint、偏好对、WildChat 缓存、blind set、预测结果或图表。复核论文现有数字需要另行提供 checkpoint/result artifact bundle；仅解压代码 ZIP 不能离线重现现有训练结果。

## 环境

### ML 环境

推荐 Linux、Python 3.10、CUDA GPU 和 24GB 显存。所有 Python 包使用 `pip` 安装，不混用 `conda install`：

```bash
conda create -n ML python=3.10 -y
conda activate ML
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

完整 SFT、候选生成、DPO 和学生模型评估按 RTX 3090 24GB 配置。小显存设备应先使用 `--limit` smoke test，并降低 batch size。

### 环境变量

```bash
cp .env.example .env
```

按所用阶段配置：

- `DOJ_PROXY`：DOJ crawler 代理，当前爬虫默认使用 `http://127.0.0.1:2778`；
- `HF_TOKEN`：下载 Qwen3/WildChat 时可能需要；
- `DASHSCOPE_API_KEY`、`ALIYUN_MODEL_NAME`、`DASHSCOPE_BASE_URL`：阿里云兼容 API；
- 或 `.env.example` 中的 GLM、Agnes、OpenAI-compatible provider 变量；
- `HF_ENDPOINT=https://hf-mirror.com`：可选 Hugging Face 镜像。

不要提交 `.env` 或真实密钥。

## 目录与阶段

```text
crawler/output/doj_raw.jsonl       crawler 工作输出，不是正式训练输入
        ↓ scripts.import_doj_raw
data/raw/doj_raw.jsonl             规范、受跟踪的 Phase 0 输入
        ↓ scripts.run_synthesis
data/synthesized/*.jsonl           SFT 与 held-out 合成数据
        ↓ scripts.run_all --only sft
checkpoints/sft/                    SFT adapter + classifier head
        ↓ scripts.generate_candidates
data/preference/candidates.jsonl
        ↓ scripts.pre_generate judge
data/preference/dpo_pairs.jsonl
        ↓ scripts.run_all --only dpo
checkpoints/dpo/                    ThreatWeaver SFT→DPO checkpoint
        ↓ scripts.run_all --only blind eval
outputs/eval/                       predictions、指标和图
        ↓ scripts.diagnose_generation
generation_diagnostics.json         无 API/GPU 的确定性生成标签诊断
```

`scripts.run_all` 只驱动 GPU 主链中的 `haystack / sft / gen_candidates / dpo / blind / eval`。DOJ 爬取、Phase 0 synthesis、API Judge 和生成端诊断必须单独执行。

## 从 DOJ 爬取开始复现

### 1. 爬取 DOJ 新闻稿

Crawler 使用独立 Python 环境，依赖见 `crawler/requirements.txt`：

```bash
python -m venv crawler
source crawler/bin/activate
pip install -r crawler/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python crawler/run.py
```

Crawler 追加写入 `crawler/output/doj_raw.jsonl`。重新全量爬取前应检查 checkpoint 与重复记录；`--fresh` 不会自动删除已有 JSONL。

课程归档已经包含规范副本 `data/raw/doj_raw.jsonl`，因此复现实验不必重新访问 DOJ。若重新爬取，使用以下命令导入：

```bash
conda run -n ML python -m scripts.import_doj_raw
```

当目标文件与源文件不同，工具会拒绝静默覆盖；确认后显式使用 `--overwrite`。导入会逐行验证 JSON，并报告 SHA-256、字节数和记录数。

### 2. Teacher 数据合成

```bash
conda run -n ML python -m scripts.run_synthesis \
  --provider aliyun --model qwen-plus --overwrite \
  --max-workers 5 --rpm 100
conda run -n ML python -m scripts.check_dedup
```

输出：

- `data/synthesized/train.jsonl`
- `data/synthesized/test.jsonl`
- `data/synthesized/hard_negatives.jsonl`

Teacher provider/model 应在新的实验 manifest 中显式记录。现有历史数据无法恢复 API provider 的精确服务端 revision。

### 3. 准备 WildChat Safe stress set

```bash
conda run -n ML python -m scripts.prepare_haystack --n 5000
```

数据源为 `allenai/WildChat-nontoxic`。当前流程将其用于 blind-set FPR stress test，不加入 SFT 训练。`nontoxic` 不等同于经人工确认无犯罪意图。

### 4. SFT 与候选生成

```bash
conda run -n ML python -m scripts.run_all --from sft --to gen_candidates \
  --batch-size 8 --gradient-accumulation-steps 2
```

SFT 使用 Qwen3-8B、4-bit NF4 QLoRA 和 pooled classifier。候选生成读取 SFT checkpoint 并写入 `data/preference/candidates.jsonl`。

### 5. API Judge 构造偏好对

```bash
conda run -n ML python -m scripts.pre_generate judge \
  --input data/preference/candidates.jsonl \
  --provider aliyun --max-workers 10
```

Judge 仅用于训练偏好对构造：同一 A/B pair 交换位置调用两次，只保留胜者语义一致的样本。最终 predictions 不使用 LLM Judge 评分。

### 6. DPO、blind set 与正式评估

```bash
conda run -n ML python -m scripts.run_all --from dpo --to eval \
  --batch-size 1 --gradient-accumulation-steps 8
```

默认比较：

- `toxic-bert`：域外 broad-toxicity reference；
- `sft-no-dpo`：SFT 消融；
- `threatweaver`：SFT→DPO 主模型。

`--only eval` 只评估已有 checkpoint，不会补跑 SFT 或 DPO。完整同名 prediction JSON 会被复用，因此改变数据或 checkpoint 后应先归档旧预测。

### 7. 无 API 的生成端诊断

```bash
conda run -n ML python -m scripts.diagnose_generation \
  outputs/eval/predictions_sft-no-dpo.json \
  outputs/eval/predictions_threatweaver.json \
  --output outputs/eval/generation_diagnostics.json
```

解析器只接受输出末尾唯一、独立的 `Threat` 或 `Safe`；其他输出记为 invalid，不做 LLM fallback。

## Smoke Test

```bash
conda run -n ML python -m pytest -q
conda run -n ML python -m scripts.run_synthesis --help
conda run -n ML python -m scripts.run_all --help
conda run -n ML python -m scripts.run_eval --help
conda run -n ML python -m scripts.diagnose_generation --help

# 有 API/GPU 时：
conda run -n ML python -m scripts.run_synthesis --provider aliyun --limit 20 --overwrite
conda run -n ML python -m scripts.run_all --from sft --to eval --limit 20 --batch-size 1
```

Smoke test 会写正式默认路径，运行前应备份已有 checkpoint、prediction 和数据文件。

## Provenance

生成公开工件 manifest：

```bash
conda run -n ML python -m scripts.build_provenance
```

默认记录配置、依赖、规范 DOJ raw 和受跟踪的合成数据。可追加本地但不分发的工件：

```bash
conda run -n ML python -m scripts.build_provenance \
  --output artifacts/provenance-local.json \
  --artifact checkpoints/sft/adapter_model.safetensors:sft_adapter:not_distributed \
  --artifact checkpoints/sft/classifier_head.pt:sft_classifier:not_distributed \
  --artifact checkpoints/dpo/adapter_model.safetensors:dpo_adapter:not_distributed \
  --artifact outputs/eval/predictions_threatweaver.json:predictions:not_distributed
```

现有 checkpoint 仅记录 `Qwen/Qwen3-8B` repo ID，没有保存当时解析到的 Hugging Face commit；Teacher/Judge 的精确服务端 revision 也未记录。因此 manifest 支持工件一致性核验，但不构成原实验的 bitwise reproducibility 证明。

## 已知限制

- SFT 分类头训练时池化 teacher-forced completion，推理时只看到 prompt；
- DPO 更新语言模型 LoRA，但不联合更新 pooled classifier；
- 生成标签指标是观察到端点分歧后的 post-hoc diagnostic；
- blind set 主要是 Teacher 合成数据，缺少独立人工 gold benchmark；
- 仅有单一基座、单 seed 和单次主要训练；
- Toxic-BERT 测量 broad toxicity，不是同任务犯罪意图模型；
- rationale 是生成输出，不证明内部推理忠实。

## 发布归档

发布前先审计 Git 跟踪清单，再创建 ZIP：

```bash
git ls-files > ../archive-manifest.txt
zip ../threatweaver-code.zip -@ < ../archive-manifest.txt
unzip -l ../threatweaver-code.zip
zipinfo -t ../threatweaver-code.zip
sha256sum ../threatweaver-code.zip
```

归档不得包含 `.env`、论文与开发文档、本地 agent 配置、checkpoint、outputs、logs 或缓存。最终应在全新目录解压并运行单元测试与 CLI `--help`。
