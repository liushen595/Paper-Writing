"""统一的 Teacher LLM 客户端封装：支持 OpenAI 兼容接口（GLM / Agnes 等）。

所有 provider 均暴露 chat(messages, **gen_kwargs) -> str 接口，
造数 / 偏好对 / Judge 复用同一套客户端。失败自动降级到下一个可用 provider。
模型名从 .env 环境变量读取，不硬编码在代码中。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests

from ..utils.env import EnvConfig, LLMProviderConfig, load_env_config
from ..utils.logging import get_logger

log = get_logger("llm_client")


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class BaseClient:
    """所有 provider 客户端的共同接口。"""

    name: str = "base"

    def __init__(self, provider: LLMProviderConfig, timeout: int = 120):
        self.provider = provider
        self.model = provider.model_name  # 从 .env 读取，不硬编码
        self.timeout = timeout

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 1024, **kw) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseClient):
    """OpenAI 兼容 chat/completions 接口（GLM、Agnes、OpenAI 中转均走此）。"""

    name = "openai_compat"

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 1024, **kw) -> str:
        url = self.provider.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kw,
        }
        headers = {"Authorization": f"Bearer {self.provider.api_key}", "Content-Type": "application/json"}
        return _retry_request(url, payload, headers, self.timeout)


def _retry_request(url: str, payload: dict, headers: dict, timeout: int, retries: int = 3, backoff: float = 2.0) -> str:
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = backoff * (2 ** i)
            log.warning(f"LLM 请求失败({i+1}/{retries}): {e}; {wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"LLM 请求重试 {retries} 次仍失败: {last_err}")


def build_client(provider_name: Optional[str] = None, env: Optional[EnvConfig] = None, model: Optional[str] = None) -> BaseClient:
    """根据 .env 中已启用的 provider 构造客户端。指定 provider_name 则用之，否则取首个可用。
    
    模型名优先级：参数 model > .env 中对应 provider 的 *_MODEL_NAME。
    """
    env = env or load_env_config()
    available = env.available_providers()
    if not available:
        raise RuntimeError("没有可用的 LLM provider，请在 .env 中配置至少一个 API_KEY 与 BASE_URL")
    if provider_name:
        providers = [p for p in available if p.name == provider_name]
        if not providers:
            raise RuntimeError(f"指定的 provider '{provider_name}' 未启用，可用: {[p.name for p in available]}")
        chosen = providers[0]
    else:
        chosen = available[0]
    if model:
        chosen.model_name = model
    return OpenAICompatibleClient(chosen)


def safe_json_extract(text: str) -> Any:
    """从 LLM 输出中抽取首个 JSON 对象/数组，自动剥离 markdown 代码块，支持截断修复。"""
    text = text.strip()
    # 剥离 ```json ... ``` 或 ``` ... ``` 包裹
    if text.startswith("```"):
        first_nl = text.find("\n")
        last_fence = text.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            text = text[first_nl + 1 : last_fence].strip()
    for start, end in (("{", "}"), ("[", "]")):
        s = text.find(start)
        if s == -1:
            continue
        e = text.rfind(end)
        if e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                # 可能是嵌套 JSON 导致 rfind 找到内层闭合括号，尝试修复
                pass
        # 截断修复：闭合括号不匹配或不存在，尝试自动补全
        return _repair_truncated_json(text[s:], start, end)
    raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}...")


def _repair_truncated_json(text: str, start: str, end: str) -> Any:
    """尝试修复被截断的 JSON：去掉尾部不完整字段，补全括号。"""
    text = text.rstrip()
    # 去掉最后一个不完整的字符串值（没有闭合引号）
    # 找最后一个未闭合的引号
    in_string = False
    escape_next = False
    last_quote_pos = -1
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            if in_string:
                last_quote_pos = i
    # 如果 in_string 为 True，说明有未闭合的字符串，截断到最后一个完整值之后
    if in_string and last_quote_pos > 0:
        # 往回找到最后一个完整的 key-value 对的逗号位置
        text = text[:last_quote_pos].rstrip()
        # 去掉尾部可能残留的逗号
        while text.endswith(","):
            text = text[:-1].rstrip()

    # 补全缺失的括号
    stack = []
    for ch in text:
        if ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch == "}" or ch == "]":
            if stack and stack[-1] == ch:
                stack.pop()
    # 追加缺失的闭合括号
    for closing in reversed(stack):
        text += closing

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"JSON 修复失败，原始文本: {text[:200]}...")
