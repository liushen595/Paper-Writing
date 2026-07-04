"""Phase 3 Stepwise Internalization（Deng et al. 2024）。

核心：从显式 CoT 模型出发，按线性调度逐步移除 CoT token 并微调。
稳定性三件套：Removal Smoothing (lambda=4)、优化器重置、左移除。

实现策略（适配 Llama-3 chat template）：
- 训练样本的 assistant 段为 "<thought>thought</thought>\n[Category: X]\nLabel"。
- thought 段被视为可移除的 CoT token 序列。
- 每移除一个 thought token：把该位置的 input_id 改为 pad/删除，label 置 -100，
  使模型在更短前缀下直接预测剩余 thought + Label。
- 移除数 s(t) = floor(delta * t / T)，加随机偏移 o ~ exp(-lambda*o)。
- 每次移除数增加时重置 AdamW 状态。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from ..data.dataset import build_train_examples
from ..models.student import StudentModel, load_tokenizer
from ..utils.config import ExperimentConfig, ImplicitCoTConfig
from ..utils.logging import get_logger, default_log_dir
from ..utils.seed import set_seed
from .sft_dataset import SFTCollator, SFTDataset, format_chat

log = get_logger("implicit_cot")


@dataclass
class RemovalState:
    """记录当前每个样本的可移除 thought 长度与已移除数。"""
    thought_len: int
    removed: int = 0


def _locate_thought_span(full_ids: list[int], thought_str: str, tokenizer) -> tuple[int, int]:
    """返回 thought 内容在 full_ids 中的 (start, end) 索引。"""
    prefix = "<thought>"
    suffix = "</thought>"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    # 简化：搜索 prefix_ids 出现位置
    start = _find_subseq(full_ids, prefix_ids)
    if start is None:
        return -1, -1
    thought_start = start + len(prefix_ids)
    end = _find_subseq(full_ids[thought_start:], suffix_ids)
    if end is None:
        return -1, -1
    return thought_start, thought_start + end


def _find_subseq(seq: list[int], sub: list[int]) -> Optional[int]:
    if not sub:
        return 0
    for i in range(len(seq) - len(sub) + 1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


def removal_schedule(t: int, T: int, delta: int) -> int:
    """线性调度 s(t) = floor(delta * t / T)。"""
    return min(delta, math.floor(delta * t / max(1, T)))


def removal_smoothing_offset(lam: float) -> int:
    """o ~ P(o) ∝ exp(-lambda * o)，非负整数。"""
    if lam == float("inf"):
        return 0
    while True:
        o = random.randint(0, 8)
        if random.random() < math.exp(-lam * o):
            return o


def apply_removal(full_ids: list[int], labels_clm: list[int], thought_span: tuple[int, int], n_remove: int, left: bool = True) -> tuple[list[int], list[int]]:
    """移除 n_remove 个 thought token（左移除：从 thought 起点移除）。返回新的 (input_ids, labels_clm)。"""
    start, end = thought_span
    if start < 0 or end <= start:
        return list(full_ids), list(labels_clm)
    thought_len = end - start
    n_remove = min(n_remove, thought_len)
    if n_remove <= 0:
        return list(full_ids), list(labels_clm)
    if left:
        keep_start = start + n_remove
    else:
        keep_start = start
        end = end - n_remove
    new_ids = full_ids[:start] + full_ids[keep_start:end] + full_ids[end:]
    new_labels = labels_clm[:start] + labels_clm[keep_start:end] + labels_clm[end:]
    return new_ids, new_labels


def reset_optimizer(opt: Optimizer) -> Optimizer:
    """重置 AdamW 的状态（一阶/二阶矩）。"""
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p, {})
            for k in ("step", "exp_avg", "exp_avg_sq"):
                if k in state:
                    state[k] = torch.zeros_like(state[k]) if torch.is_tensor(state[k]) else 0
    return opt


def train_implicit_cot(cfg: ExperimentConfig, ic_cfg: Optional[ImplicitCoTConfig] = None) -> Path:
    ic_cfg = ic_cfg or cfg.implicit_cot
    set_seed(cfg.seed)
    log.info(f"=== Phase 3 Stepwise Internalization | delta={ic_cfg.delta_per_epoch} lambda={ic_cfg.lambda_smoothing} ===")

    tokenizer = load_tokenizer(cfg.sft.base_model)
    examples = build_train_examples(cfg.data, split="train")
    if not examples:
        raise RuntimeError("无训练样本，请先运行 synthesis")

    # 预计算每个样本的 thought span
    spans: list[tuple[int, int]] = []
    base_cache: list[dict] = []
    for ex in examples:
        full, _ = format_chat(ex, tokenizer)
        full_ids = tokenizer(full, truncation=True, max_length=ic_cfg.max_seq_len, add_special_tokens=False)["input_ids"]
        labels_clm = list(full_ids)
        # 简化：prompt 段（含 <thought> 之前）置 -100；这里复用 SFTDataset 逻辑的话需要 prompt_ids
        span = _locate_thought_span(full_ids, ex.thought_process, tokenizer)
        spans.append(span)
        base_cache.append({"input_ids": full_ids, "labels_clm": labels_clm, "labels_cls": 1 if ex.label == "Threat" else 0})

    model = StudentModel.load(cfg.sft, ic_cfg.sft_ckpt)
    model.train()
    device = next(model.parameters()).device
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=ic_cfg.learning_rate)

    out_dir = Path(ic_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collator = SFTCollator(tokenizer, max_seq_len=ic_cfg.max_seq_len)
    T = len(base_cache) * ic_cfg.num_epochs  # 总步数近似
    global_step = 0
    prev_s = 0
    for epoch in range(ic_cfg.num_epochs):
        epoch_indices = list(range(len(base_cache)))
        random.shuffle(epoch_indices)
        for idx in epoch_indices:
            t = global_step
            s = removal_schedule(t, T, ic_cfg.delta_per_epoch)
            if ic_cfg.left_removal:
                s += removal_smoothing_offset(ic_cfg.lambda_smoothing)
            if s > prev_s and ic_cfg.reset_optimizer_on_removal:
                optimizer = reset_optimizer(optimizer)
                log.info(f"step={t} 移除数 {prev_s}->{s}, 优化器已重置")
                prev_s = s
            base = base_cache[idx]
            new_ids, new_labels = apply_removal(base["input_ids"], base["labels_clm"], spans[idx], s, left=ic_cfg.left_removal)
            batch = collator([{"input_ids": new_ids, "labels_clm": new_labels, "labels_cls": base["labels_cls"]}])
            batch = {k: v.to(device) for k, v in batch.__dict__.items()}
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels_clm=batch["labels_clm"], labels_cls=batch["labels_cls"],
            )
            loss = outputs["clm_loss"] + outputs["cls_loss"]
            loss.backward()
            if (global_step + 1) % ic_cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % 50 == 0:
                log.info(f"epoch={epoch} step={global_step} s={s} loss={loss.item():.4f}")
        log.info(f"epoch {epoch} 完成, 当前移除数 s={prev_s}")
    model.save(out_dir)
    log.info(f"Stepwise Internalization 完成, 模型保存到 {out_dir}")
    return out_dir
