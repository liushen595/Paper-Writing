## Copyright [2026] [Yijun Liu, Soochow University]
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

"""环境变量加载：从项目根 .env 读取，供 Teacher LLM / Judge / HF 调用使用。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv 未安装时的兜底
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"


def setup_hf_mirror(use_mirror: bool) -> None:
    """设置 HuggingFace 镜像站环境变量。

    必须在任何 HF 库（transformers, datasets, huggingface_hub）导入之前调用。
    """
    if use_mirror:
        os.environ.setdefault("HF_ENDPOINT", HF_MIRROR_ENDPOINT)


def _load_env() -> None:
    env_path = PROJECT_ROOT / ".env"
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=False)


@dataclass
class LLMProviderConfig:
    name: str
    api_key: str
    base_url: str
    model_name: str  # 从 .env 读取，不硬编码

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)


@dataclass
class EnvConfig:
    glm: LLMProviderConfig
    agnes: LLMProviderConfig
    aliyun: LLMProviderConfig
    openai: LLMProviderConfig
    hf_token: Optional[str]
    wandb_api_key: Optional[str]
    wandb_project: str
    data_dir: Path
    raw_dir: Path

    def available_providers(self) -> list[LLMProviderConfig]:
        return [p for p in (self.glm, self.agnes, self.aliyun, self.openai) if p.enabled]


def load_env_config() -> EnvConfig:
    _load_env()
    data_dir = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()
    raw_dir = Path(os.environ.get("RAW_DIR", PROJECT_ROOT / "crawler" / "output")).resolve()
    return EnvConfig(
        glm=LLMProviderConfig(
            name="glm",
            api_key=os.environ.get("GLM_API_KEY", "").strip(),
            base_url=os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/").strip(),
            model_name=os.environ.get("GLM_MODEL_NAME", "qwen-plus").strip(),
        ),
        agnes=LLMProviderConfig(
            name="agnes",
            api_key=os.environ.get("AGNES_API_KEY", "").strip(),
            base_url=os.environ.get("AGNES_BASE_URL", "").strip(),
            model_name=os.environ.get("AGNES_MODEL_NAME", "agnes-2.0-flash").strip(),
        ),
        aliyun=LLMProviderConfig(
            name="aliyun",
            api_key=os.environ.get("DASHSCOPE_API_KEY", "").strip(),
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
            model_name=os.environ.get("ALIYUN_MODEL_NAME", "qwen-plus").strip(),
        ),
        openai=LLMProviderConfig(
            name="openai",
            api_key=os.environ.get("OPENAI_API_KEY", "").strip(),
            base_url=os.environ.get("OPENAI_BASE_URL", "").strip(),
            model_name=os.environ.get("OPENAI_MODEL_NAME", "gpt-4o-mini").strip(),
        ),
        hf_token=(os.environ.get("HF_TOKEN") or None),
        wandb_api_key=(os.environ.get("WANDB_API_KEY") or None),
        wandb_project=os.environ.get("WANDB_PROJECT", "criminal-intent"),
        data_dir=data_dir,
        raw_dir=raw_dir,
    )
