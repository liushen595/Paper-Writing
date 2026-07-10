"""全局可复现性：seed 设置。"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # TF32：Ampere+ GPU 上 matmul/cudnn 用 TF32 替代 fp32，显著加速且精度损失可忽略
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        pass
