"""Phase 1 SFT 训练循环：联合损失 alpha*L_cls + beta*L_clm。

支持断点续训：checkpoint 按 epoch 编号保存，resume 时自动从最新 checkpoint 继续。
num_epochs 为总轮数，resume 时跳过已完成的 epoch。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ..data.dataset import build_train_examples
from ..models.student import StudentModel, load_tokenizer
from ..utils.config import ExperimentConfig, SFTConfig
from ..utils.logging import get_logger, default_log_dir
from ..utils.seed import set_seed
from .sft_dataset import SFTCollator, SFTDataset

log = get_logger("sft_train")


def _find_latest_checkpoint(ckpt_dir: Path) -> tuple[Optional[Path], int]:
    """找到最新的 checkpoint-XXXX 目录，返回 (路径, 已完成 epoch 数)。"""
    if not ckpt_dir.exists():
        return None, 0
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if not ckpts:
        return None, 0
    latest = ckpts[-1]
    completed_epochs = int(latest.name.split("-")[-1])
    return latest, completed_epochs


def train_sft(cfg: ExperimentConfig, sft_cfg: Optional[SFTConfig] = None, split: str = "train") -> Path:
    sft_cfg = sft_cfg or cfg.sft
    set_seed(cfg.seed)
    log_dir = default_log_dir()
    log.info(f"=== Phase 1 SFT | base={sft_cfg.base_model} | seed={cfg.seed} | epochs={sft_cfg.num_epochs} ===")

    tokenizer = load_tokenizer(sft_cfg.base_model)
    examples = build_train_examples(cfg.data, split=split)
    if not examples:
        raise RuntimeError(f"无训练样本，请先运行 src/data/synthesis.py 生成数据 (split={split})")
    dataset = SFTDataset(examples, tokenizer, max_seq_len=sft_cfg.max_seq_len)
    collator = SFTCollator(tokenizer, max_seq_len=sft_cfg.max_seq_len)
    loader = DataLoader(
        dataset, batch_size=sft_cfg.per_device_batch_size, shuffle=True,
        collate_fn=collator, num_workers=2, pin_memory=True,
    )

    out_dir = Path(sft_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训：查找最新 checkpoint
    latest_ckpt, start_epoch = _find_latest_checkpoint(out_dir)
    if latest_ckpt:
        log.info(f"发现 checkpoint: {latest_ckpt}，从 epoch {start_epoch} 继续训练（总轮数 {sft_cfg.num_epochs}）")
        model = StudentModel.load(sft_cfg, latest_ckpt)
    else:
        log.info("无已有 checkpoint，从头训练")
        model = StudentModel(sft_cfg)

    if start_epoch >= sft_cfg.num_epochs:
        log.info(f"已完成 {start_epoch} 轮 >= 目标 {sft_cfg.num_epochs} 轮，跳过训练")
        return out_dir

    model.train()
    device = next(model.parameters()).device
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=sft_cfg.learning_rate)

    steps_per_epoch = len(loader)
    total_steps = sft_cfg.num_epochs * steps_per_epoch
    warmup = int(sft_cfg.warmup_ratio * total_steps)
    global_step = start_epoch * steps_per_epoch

    for epoch in range(start_epoch, sft_cfg.num_epochs):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.__dict__.items()}
            outputs = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels_clm=batch["labels_clm"], labels_cls=batch["labels_cls"],
            )
            loss = sft_cfg.cls_loss_weight * outputs["cls_loss"] + sft_cfg.clm_loss_weight * outputs["clm_loss"]
            if global_step < warmup:
                lr_scale = (global_step + 1) / max(1, warmup)
                for pg in optimizer.param_groups:
                    pg["lr"] = sft_cfg.learning_rate * lr_scale
            loss.backward()
            if (global_step + 1) % sft_cfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            if global_step % 20 == 0:
                log.info(f"epoch={epoch} step={global_step}/{total_steps} loss={loss.item():.4f} "
                         f"cls={outputs['cls_loss'].item():.4f} clm={outputs['clm_loss'].item():.4f}")

        # 每个 epoch 结束保存 checkpoint
        ckpt_path = out_dir / f"checkpoint-{epoch + 1}"
        model.save(ckpt_path)
        log.info(f"epoch {epoch} 完成, checkpoint 保存到 {ckpt_path}")

    # 最终保存为 adapter_model 目录（兼容 PeftModel.from_pretrained）
    model.save(out_dir)
    log.info(f"SFT 训练完成, 最终模型保存到 {out_dir}")
    return out_dir
