"""集中式配置：基于 YAML + dataclass，所有可调超参在此统一管理。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

from .env import PROJECT_ROOT


@dataclass
class DataConfig:
    raw_criminal: str = "crawler/output/doj_criminal.jsonl"
    synthesized_dir: str = "data/synthesized"
    preference_dir: str = "data/preference"
    blind_dir: str = "data/blind"
    train_ratio: float = 0.8
    seed: int = 42
    max_text_len: int = 512


@dataclass
class SFTConfig:
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    cls_loss_weight: float = 1.0   # alpha
    clm_loss_weight: float = 1.0   # beta
    use_roberta_distill: bool = False
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    max_seq_len: int = 1024
    output_dir: str = "checkpoints/sft"


@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 5e-7
    num_epochs: int = 1
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_prompt_len: int = 256
    max_length: int = 1024
    output_dir: str = "checkpoints/dpo"


@dataclass
class ImplicitCoTConfig:
    sft_ckpt: str = "checkpoints/sft"
    delta_per_epoch: int = 8
    lambda_smoothing: float = 4.0
    reset_optimizer_on_removal: bool = True
    left_removal: bool = True
    learning_rate: float = 1e-5
    num_epochs: int = 20
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_len: int = 1024
    output_dir: str = "checkpoints/implicit_cot"


@dataclass
class EvalConfig:
    blind_csv: str = "data/blind/test_blind.csv"
    threshold: float = 0.5
    baselines: list[str] = field(
        default_factory=lambda: ["toxic-bert", "llama3-zeroshot", "explicit-cot", "sft-no-dpo", "roberta-large"]
    )
    judge_provider: str = "glm"
    judge_swap_positions: bool = True
    judge_reference_guided: bool = True
    output_dir: str = "outputs/eval"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)
    implicit_cot: ImplicitCoTConfig = field(default_factory=ImplicitCoTConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42


def _resolve_paths(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    """把以 checkpoints/ data/ outputs/ 开头的相对路径解析为项目根绝对路径。"""
    prefixes = ("checkpoints/", "data/", "outputs/", "crawler/")
    for section in cfg_dict.values():
        if not isinstance(section, dict):
            continue
        for k, v in section.items():
            if isinstance(v, str) and v.split("/", 1)[0] + "/" in prefixes:
                section[k] = str(PROJECT_ROOT / v)
    return cfg_dict


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    raw = _resolve_paths(raw)
    return _dict_to_config(raw)


def _dict_to_config(d: dict[str, Any]) -> ExperimentConfig:
    cfg = ExperimentConfig()
    for section_name in ("data", "sft", "dpo", "implicit_cot", "eval"):
        section = d.get(section_name)
        if not section:
            continue
        sub = getattr(cfg, section_name)
        for k, v in section.items():
            if hasattr(sub, k):
                setattr(sub, k, v)
    if "seed" in d:
        cfg.seed = int(d["seed"])
        cfg.data.seed = cfg.seed
    return cfg


def save_config(cfg: ExperimentConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)


def default_config() -> ExperimentConfig:
    return ExperimentConfig()
