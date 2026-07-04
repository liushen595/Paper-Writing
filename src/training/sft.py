"""Phase 1 SFT 训练循环：联合损失 alpha*L_cls + beta*L_clm。

注：本文件仅定义训练函数；运行入口在 scripts/run_sft.py，本任务不执行训练。
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


def train_sft(cfg: ExperimentConfig, sft_cfg: Optional[SFTConfig] = None, split: str = "train") -> Path:
    sft_cfg = sft_cfg or cfg.sft
    set_seed(cfg.seed)
    log_dir = default_log_dir()
    log.info(f"=== Phase 1 SFT 开始 | base={sft_cfg.base_model} | seed={cfg.seed} ===")

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

    model = StudentModel(sft_cfg)
    model.train()
    device = next(model.parameters()).device
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=sft_cfg.learning_rate)

    total_steps = sft_cfg.num_epochs * len(loader)
    warmup = int(sft_cfg.warmup_ratio * total_steps)
    out_dir = Path(sft_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    for epoch in range(sft_cfg.num_epochs):
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
        log.info(f"epoch {epoch} 完成")
    model.save(out_dir)
    log.info(f"SFT 训练完成, 模型保存到 {out_dir}")
    return out_dir
