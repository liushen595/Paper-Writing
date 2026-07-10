"""Phase 3 Stepwise Internalization（Deng et al. 2024）。

支持断点续训与早停：checkpoint 按 epoch 编号，resume 时恢复 removal state。
num_epochs 为总轮数。

核心：从显式 CoT 模型出发，按线性调度逐步移除 CoT token 并微调。
稳定性三件套：Removal Smoothing (lambda=4)、优化器重置、左移除。
早停基于验证集 loss，连续 patience 个 epoch 无改善则终止。
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import build_train_examples
from ..models.student import StudentModel, load_tokenizer
from ..utils.config import ExperimentConfig, ImplicitCoTConfig
from ..utils.logging import get_logger, default_log_dir
from ..utils.seed import set_seed
from .sft_dataset import SFTCollator, format_chat

log = get_logger("implicit_cot")


def _locate_thought_span(full_ids: list[int], tokenizer) -> tuple[int, int]:
    """返回 <thought>...</thought> 内容在 full_ids 中的 (start, end) 索引。"""
    prefix_ids = tokenizer("<thought>", add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer("</thought>", add_special_tokens=False)["input_ids"]
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
    return min(delta, math.floor(delta * t / max(1, T)))


def removal_smoothing_offset(lam: float) -> int:
    if lam == float("inf"):
        return 0
    for _ in range(20):
        o = random.randint(0, 8)
        if random.random() < math.exp(-lam * o):
            return o
    return 0


def apply_removal(full_ids: list[int], labels_clm: list[int], thought_span: tuple[int, int], n_remove: int, left: bool = True) -> tuple[list[int], list[int]]:
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
    for group in opt.param_groups:
        for p in group["params"]:
            state = opt.state.get(p, {})
            for k in ("step", "exp_avg", "exp_avg_sq"):
                if k in state:
                    state[k] = torch.zeros_like(state[k]) if torch.is_tensor(state[k]) else 0
    return opt


def _save_ckpt_state(out_dir: Path, epoch: int, removed_so_far: int):
    state = {"epoch": epoch, "removed_so_far": removed_so_far}
    with open(out_dir / "train_state.json", "w") as f:
        json.dump(state, f)


def _load_ckpt_state(out_dir: Path) -> tuple[int, int]:
    p = out_dir / "train_state.json"
    if p.exists():
        with open(p, "r") as f:
            state = json.load(f)
        return state.get("epoch", 0), state.get("removed_so_far", 0)
    return 0, 0


def _build_val_cache(val_examples, tokenizer, max_seq_len):
    """构建验证集缓存。"""
    cache = []
    spans = []
    for ex in val_examples:
        full, _ = format_chat(ex, tokenizer)
        full_ids = tokenizer(full, truncation=True, max_length=max_seq_len, add_special_tokens=False)["input_ids"]
        labels_clm = list(full_ids)
        span = _locate_thought_span(full_ids, tokenizer)
        spans.append(span)
        cache.append({"input_ids": full_ids, "labels_clm": labels_clm, "labels_cls": 1 if ex.label == "Threat" else 0})
    return cache, spans


def _evaluate_implicit_cot(model, val_cache, val_spans, collator, device, n_remove):
    """在验证集上计算平均 loss，使用当前 removal level。"""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for i, base in enumerate(tqdm(val_cache, desc="Validate", unit="sample", leave=False)):
            new_ids, new_labels = apply_removal(base["input_ids"], base["labels_clm"], val_spans[i], n_remove, left=True)
            batch = collator([{"input_ids": new_ids, "labels_clm": new_labels, "labels_cls": base["labels_cls"]}])
            batch = {k: v.to(device) for k, v in batch.__dict__.items()}
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels_clm=batch["labels_clm"], labels_cls=batch["labels_cls"],
            )
            total_loss += outputs["clm_loss"].item() + outputs["cls_loss"].item()
            n_batches += 1
    model.train()
    return total_loss / max(1, n_batches)


def _find_latest_checkpoint(ckpt_dir: Path) -> tuple[Optional[Path], int]:
    if not ckpt_dir.exists():
        return None, 0
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    completed_epochs = int(latest.name.split("-")[-1])
    return latest, completed_epochs


def train_implicit_cot(cfg: ExperimentConfig, ic_cfg: Optional[ImplicitCoTConfig] = None) -> Path:
    ic_cfg = ic_cfg or cfg.implicit_cot
    set_seed(cfg.seed)
    log.info(f"=== Phase 3 Stepwise Internalization | delta={ic_cfg.delta_per_epoch} lambda={ic_cfg.lambda_smoothing} | epochs={ic_cfg.num_epochs} ===")

    tokenizer = load_tokenizer(cfg.sft.base_model)
    examples = build_train_examples(cfg.data, split="train")
    if not examples:
        raise RuntimeError("无训练样本，请先运行 synthesis")

    # 预计算每个样本的 thought span
    base_cache: list[dict] = []
    spans: list[tuple[int, int]] = []
    for ex in examples:
        full, _ = format_chat(ex, tokenizer)
        full_ids = tokenizer(full, truncation=True, max_length=ic_cfg.max_seq_len, add_special_tokens=False)["input_ids"]
        labels_clm = list(full_ids)
        span = _locate_thought_span(full_ids, tokenizer)
        spans.append(span)
        base_cache.append({"input_ids": full_ids, "labels_clm": labels_clm, "labels_cls": 1 if ex.label == "Threat" else 0})

    out_dir = Path(ic_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 验证集
    val_examples = build_train_examples(cfg.data, split="test")
    val_cache, val_spans = _build_val_cache(val_examples, tokenizer, ic_cfg.max_seq_len)

    # 断点续训
    latest_ckpt, start_epoch = _find_latest_checkpoint(out_dir)
    removed_so_far = 0
    if latest_ckpt:
        _, removed_so_far = _load_ckpt_state(latest_ckpt)
        log.info(f"发现 checkpoint: {latest_ckpt}, 从 epoch {start_epoch} 继续, 已移除 {removed_so_far} tokens（总轮数 {ic_cfg.num_epochs}）")
        model = StudentModel.load(cfg.sft, latest_ckpt)
    else:
        log.info("无已有 checkpoint，从头训练")
        model = StudentModel.load(cfg.sft, ic_cfg.sft_ckpt)

    if start_epoch >= ic_cfg.num_epochs:
        log.info(f"已完成 {start_epoch} 轮 >= 目标 {ic_cfg.num_epochs} 轮，跳过训练")
        return out_dir

    model.train()
    device = next(model.parameters()).device
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=ic_cfg.learning_rate)

    collator = SFTCollator(tokenizer, max_seq_len=ic_cfg.max_seq_len)
    T = len(base_cache) * ic_cfg.num_epochs
    global_step = start_epoch * len(base_cache)
    prev_s = removed_so_far

    best_val_loss = float("inf")
    patience_counter = 0
    patience = ic_cfg.early_stopping_patience
    min_delta = ic_cfg.early_stopping_min_delta

    for epoch in range(start_epoch, ic_cfg.num_epochs):
        epoch_indices = list(range(len(base_cache)))
        random.shuffle(epoch_indices)
        pbar = tqdm(epoch_indices, desc=f"ImplicitCoT epoch {epoch+1}/{ic_cfg.num_epochs}", unit="sample")
        for idx in pbar:
            t = global_step
            s = removal_schedule(t, T, ic_cfg.delta_per_epoch) + removal_smoothing_offset(ic_cfg.lambda_smoothing)
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
            pbar.set_postfix(s=s, loss=f"{loss.item():.4f}")

        # 每个 epoch 结束保存 checkpoint
        ckpt_path = out_dir / f"checkpoint-{epoch + 1}"
        model.save(ckpt_path)
        _save_ckpt_state(ckpt_path, epoch + 1, prev_s)
        log.info(f"epoch {epoch} 完成, checkpoint 保存到 {ckpt_path}, 当前移除数 s={prev_s}")

        # 验证 + 早停
        val_loss = _evaluate_implicit_cot(model, val_cache, val_spans, collator, device, prev_s)
        log.info(f"epoch {epoch} 验证 loss={val_loss:.4f} (best={best_val_loss:.4f}, patience={patience_counter}/{patience})")
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = out_dir / "best"
            model.save(best_path)
            _save_ckpt_state(best_path, epoch + 1, prev_s)
            log.info(f"验证 loss 改善, 保存 best 模型到 {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"早停触发: 连续 {patience} 个 epoch 验证 loss 无改善, 终止训练")
                break

    model.save(out_dir)
    log.info(f"Stepwise Internalization 完成, 模型保存到 {out_dir}")
    return out_dir
