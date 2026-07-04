"""分类头独立模块（与 student.py 中的 ClassifierHead 共享设计，便于单独训练 RoBERTa Teacher）。"""
from __future__ import annotations

import torch
from torch import nn


class PooledClassifier(nn.Module):
    """通用 mean-pool 分类头，可挂载到 RoBERTa/BERT 之上做 Teacher Classifier。"""

    def __init__(self, hidden_size: int, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return self.linear(self.dropout(pooled))


def build_roberta_teacher(model_name: str = "roberta-large", num_labels: int = 2) -> tuple[nn.Module, nn.Module]:
    """构建 RoBERTa-Large + PooledClassifier，作为 ToXCL 蒸馏源 Teacher。"""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    backbone = AutoModel.from_pretrained(model_name)
    head = PooledClassifier(backbone.config.hidden_size, num_labels=num_labels)
    return nn.Sequential(backbone, head), tok
