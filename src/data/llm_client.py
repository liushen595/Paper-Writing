"""统一的 Teacher LLM 客户端封装：支持 GLM / Gemini / OpenAI 兼容接口。

所有 provider 均暴露 chat(messages, **gen_kwargs) -> str 接口，
造数 / 偏好对 / Judge 复用同一套客户端。失败自动降级到下一个可用 provider。
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

    def __init__(self, provider: LLMProviderConfig, model: str, timeout: int = 120):
        self.provider = provider
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 1024, **kw) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseClient):
    """OpenAI 兼容 chat/completions 接口（GLM、OpenAI 中转均走此）。"""

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


class GeminiClient(BaseClient):
    """Google Gemini generateContent 接口。"""

    name = "gemini"

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 1024, **kw) -> str:
        # Gemini 把 system 作为 systemInstruction，user/model 进 contents
        system_msgs = [m for m in messages if m.role == "system"]
        contents = [
            {"role": m.role if m.role in ("user", "model") else "user", "parts": [{"text": m.content}]}
            for m in messages if m.role != "system"
        ]
        body: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens, **kw},
        }
        if system_msgs:
            body["systemInstruction"] = {"parts": [{"text": " ".join(m.content for m in system_msgs)}]}
        url = f"{self.provider.base_url.rstrip('/')}/models/{self.model}:generateContent?key={self.provider.api_key}"
        resp = requests.post(url, json=body, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


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


# --- provider -> (client_class, default_model) 映射 ---
_PROVIDER_REGISTRY = {
    "glm": (OpenAICompatibleClient, "glm-4-flash"),
    "openai": (OpenAICompatibleClient, "gpt-4o-mini"),
    "gemini": (GeminiClient, "gemini-2.0-flash"),
}


def build_client(provider_name: Optional[str] = None, env: Optional[EnvConfig] = None, model: Optional[str] = None) -> BaseClient:
    """根据 .env 中已启用的 provider 构造客户端。指定 provider_name 则用之，否则取首个可用。"""
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
    cls, default_model = _PROVIDER_REGISTRY[chosen.name]
    return cls(chosen, model=model or default_model)


def safe_json_extract(text: str) -> Any:
    """从 LLM 输出中抽取首个 JSON 对象/数组。"""
    text = text.strip()
    for start, end in (("{", "}"), ("[", "]")):
        s = text.find(start)
        e = text.rfind(end)
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(text[s : e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}...")
