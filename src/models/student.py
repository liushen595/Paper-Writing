"""Qwen3-8B + QLoRA 学生模型加载，附 ToXCL 风格分类头。

设计要点（基于 Plan §2.2 / ToXCL Hoang et al. 2024）：
- 基座 4-bit NF4 量化 + LoRA 适配器（target=q_proj,v_proj）。
- 在最后隐藏层之上加 mean-pool 分类头（二分类 Threat/Safe）。
- CLM head 复用基座 lm_head，对 thought_process + label 计算损失。
- 联合损失: alpha * L_cls + beta * L_clm。
- 推理时 Conditional Decoding Constraint：cls=Safe 直接输出 [None]。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from ..utils.config import SFTConfig
from ..utils.logging import get_logger

log = get_logger("student_model")


@dataclass
class StudentOutput:
    logits: torch.Tensor              # (B, T, V) CLM logits
    cls_logits: torch.Tensor          # (B, 2) 分类头 logits
    hidden_states: Optional[torch.Tensor] = None


class ClassifierHead(nn.Module):
    """ToXCL 风格：mean-pool last hidden state -> linear -> 2 类。"""

    def __init__(self, hidden_size: int, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.linear(self.dropout(pooled))


class StudentModel(nn.Module):
    """QLoRA 基座 + 分类头的封装。"""

    def __init__(self, sft_cfg: SFTConfig):
        super().__init__()
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=sft_cfg.quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=sft_cfg.double_quant,
        )
        log.info(f"加载基座模型 {sft_cfg.base_model} (4-bit NF4)")
        base = AutoModelForCausalLM.from_pretrained(
            sft_cfg.base_model, quantization_config=bnb_config, device_map="auto",
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

        lora_cfg = LoraConfig(
            r=sft_cfg.lora_r, lora_alpha=sft_cfg.lora_alpha, lora_dropout=sft_cfg.lora_dropout,
            target_modules=sft_cfg.target_modules, bias="none", task_type="CAUSAL_LM",
        )
        self.base = get_peft_model(base, lora_cfg)
        self.hidden_size = base.config.hidden_size
        self.classifier = ClassifierHead(self.hidden_size)
        self.classifier.to(base.dtype)
        log.info(f"Student 模型就绪: hidden_size={self.hidden_size}, LoRA r={sft_cfg.lora_r}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels_clm: Optional[torch.Tensor] = None,
        labels_cls: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> dict:
        out = self.base(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        last_hidden = out.hidden_states[-1]
        cls_logits = self.classifier(last_hidden, attention_mask)
        result: dict = {"logits": out.logits, "cls_logits": cls_logits, "hidden_states": last_hidden}
        if labels_clm is not None:
            result["clm_loss"] = out.loss if out.loss is not None else _causal_lm_loss(out.logits, labels_clm)
        if labels_cls is not None:
            loss_fct = nn.CrossEntropyLoss()
            result["cls_loss"] = loss_fct(cls_logits, labels_cls)
        return result

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.base.save_pretrained(out_dir)
        torch.save(self.classifier.state_dict(), out_dir / "classifier_head.pt")
        log.info(f"Student 模型保存到 {out_dir}")

    @classmethod
    def load(cls, sft_cfg: SFTConfig, ckpt_dir: str | Path) -> "StudentModel":
        """加载 StudentModel。ckpt_dir 里应有 adapter_config.json + adapter_model.* + classifier_head.pt。

        SFT 与 DPO checkpoint 都用本方法加载：DPO 训练时已把 SFT 的 classifier_head.pt 复制到 DPO 输出目录，
        DPO 的 LoRA adapter 直接通过 PeftModel.from_pretrained 套到基座上。
        """
        import os
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        ckpt_dir = Path(ckpt_dir)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=sft_cfg.quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=sft_cfg.double_quant,
        )
        log.info(f"加载基座模型 {sft_cfg.base_model} (4-bit NF4) for ckpt {ckpt_dir}")
        base = AutoModelForCausalLM.from_pretrained(
            sft_cfg.base_model, quantization_config=bnb_config, device_map="auto",
        )
        if (ckpt_dir / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(base, str(ckpt_dir))
            log.info(f"已加载 LoRA adapter: {ckpt_dir}")
        else:
            log.warning(f"未找到 adapter_config.json: {ckpt_dir}; 使用未微调基座")
            model = base

        # 包装成 StudentModel 以复用分类头逻辑
        obj = cls.__new__(cls)
        torch.nn.Module.__init__(obj)
        obj.base = model
        obj.hidden_size = base.config.hidden_size
        obj.classifier = ClassifierHead(obj.hidden_size)
        obj.classifier.to(base.dtype)
        cls_head_path = ckpt_dir / "classifier_head.pt"
        if cls_head_path.exists():
            state = torch.load(cls_head_path, map_location="cpu")
            obj.classifier.load_state_dict(state)
            log.info(f"分类头加载自 {cls_head_path}")
        else:
            log.warning(f"分类头不存在: {cls_head_path}; 使用随机初始化")
        return obj


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    return loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def load_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
