"""Phase 2 DPO 训练：基于 trl.DPOTrainer + QLoRA。

支持断点续训：从最新 checkpoint 恢复。
DPO beta 起始 0.1（来自 Wen et al. 2023 KL 系数经验甜点）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..data.preference import load_preference_pairs
from ..utils.config import DPOConfig, ExperimentConfig
from ..utils.env import PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.seed import set_seed

log = get_logger("dpo_train")


def _find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """找到最新的 checkpoint-XXXX 目录。"""
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return ckpts[-1] if ckpts else None


def build_dpo_dataset(cfg: ExperimentConfig):
    """从 dpo_pairs.jsonl 构造 trl DPO 所需的 Dataset。"""
    from datasets import Dataset
    pairs_path = (PROJECT_ROOT / cfg.data.preference_dir / "dpo_pairs.jsonl").resolve()
    if not pairs_path.exists():
        raise RuntimeError(f"偏好对数据不存在: {pairs_path}; 请先运行 src/data/preference.py")
    pairs = load_preference_pairs(pairs_path)
    log.info(f"加载 DPO 偏好对: {len(pairs)} 条")
    return Dataset.from_list([
        {"prompt": p["prompt"], "chosen": p["chosen"], "rejected": p["rejected"]}
        for p in pairs
    ])


def train_dpo(cfg: ExperimentConfig, dpo_cfg: Optional[DPOConfig] = None) -> Path:
    dpo_cfg = dpo_cfg or cfg.dpo
    set_seed(cfg.seed)
    log.info(f"=== Phase 2 DPO | beta={dpo_cfg.beta} | seed={cfg.seed} | epochs={dpo_cfg.num_epochs} ===")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    from peft import LoraConfig
    from trl import DPOConfig as TRLDPOConfig, DPOTrainer

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(cfg.sft.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_ckpt = Path(cfg.sft.output_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.sft.base_model, quantization_config=bnb, device_map="auto",
    )
    if (sft_ckpt / "adapter_config.json").exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(sft_ckpt))
        log.info(f"已加载 SFT LoRA 权重: {sft_ckpt}")

    lora_cfg = LoraConfig(
        r=cfg.sft.lora_r, lora_alpha=cfg.sft.lora_alpha, lora_dropout=cfg.sft.lora_dropout,
        target_modules=cfg.sft.target_modules, bias="none", task_type="CAUSAL_LM",
    )

    dataset = build_dpo_dataset(cfg)
    out_dir = Path(dpo_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训
    resume_ckpt = _find_latest_checkpoint(out_dir)

    trl_cfg = TRLDPOConfig(
        beta=dpo_cfg.beta,
        learning_rate=dpo_cfg.learning_rate,
        num_train_epochs=dpo_cfg.num_epochs,
        per_device_train_batch_size=dpo_cfg.per_device_batch_size,
        gradient_accumulation_steps=dpo_cfg.gradient_accumulation_steps,
        max_prompt_length=dpo_cfg.max_prompt_len,
        max_length=dpo_cfg.max_length,
        output_dir=str(out_dir),
        save_strategy="epoch",
        logging_steps=20,
        bf16=True,
        gradient_checkpointing=True,
        peft_config=lora_cfg,
    )
    trainer = DPOTrainer(
        model=model, args=trl_cfg, train_dataset=dataset, processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
    trainer.save_model(str(out_dir))

    # 保留 SFT 阶段的分类头，使 dpo-only baseline 可加载完整 StudentModel
    import shutil
    sft_cls_head = sft_ckpt / "classifier_head.pt"
    if sft_cls_head.exists():
        shutil.copy2(sft_cls_head, out_dir / "classifier_head.pt")
        log.info(f"分类头已复制到 DPO 输出目录: {sft_cls_head} -> {out_dir / 'classifier_head.pt'}")
    else:
        log.warning(f"SFT 分类头不存在: {sft_cls_head}; dpo-only baseline 将使用随机初始化的分类头")

    log.info(f"DPO 训练完成, 模型保存到 {out_dir}")
    return out_dir
