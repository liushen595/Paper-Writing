"""统一的 Teacher LLM 客户端封装：支持 OpenAI 兼容接口（GLM / Agnes 等），流式传输。

所有 provider 均暴露 chat(messages, **gen_kwargs) -> str 接口，
造数 / 偏好对 / Judge 复用同一套客户端。失败自动重试。
模型名从 .env 环境变量读取，不硬编码在代码中。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from openai import OpenAI

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

    def __init__(self, provider: LLMProviderConfig, timeout: int = 180):
        self.provider = provider
        self.model = provider.model_name
        self.timeout = timeout

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 2048, **kw) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(BaseClient):
    """OpenAI 兼容 chat/completions 接口，流式传输。"""

    name = "openai_compat"

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 2048, **kw) -> str:
        url = self.provider.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            **kw,
        }
        headers = {"Authorization": f"Bearer {self.provider.api_key}", "Content-Type": "application/json"}
        return _stream_request(url, payload, headers, self.timeout)


class AliyunClient(BaseClient):
    """阿里云通义千问客户端：OpenAI SDK + 流式 + 结构化 JSON 输出。"""

    name = "aliyun"

    def __init__(self, provider: LLMProviderConfig, timeout: int = 180):
        super().__init__(provider, timeout)
        self._client = OpenAI(api_key=provider.api_key, base_url=provider.base_url)

    def chat(self, messages: list[ChatMessage], temperature: float = 0.7, max_tokens: int = 2048, **kw) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            response_format={"type": "json_object"},
            **kw,
        )
        chunks: list[str] = []
        received_done = False
        last_log_len = 0
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                chunks.append(delta.content)
            current_len = sum(len(c) for c in chunks)
            if current_len - last_log_len >= 200:
                log.debug(f"流式接收中... 已接收 {current_len} 字符")
                last_log_len = current_len
            if chunk.choices[0].finish_reason is not None:
                received_done = True
        if not received_done:
            partial = "".join(chunks)
            raise RuntimeError(f"流式响应被截断，已接收 {len(partial)} 字符")
        return "".join(chunks)


def _stream_request(url: str, payload: dict, headers: dict, timeout: int) -> str:
    """流式请求：逐 chunk 接收，拼接完整响应。检测截断并抛异常以触发重试。"""
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    chunks: list[str] = []
    received_done = False
    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                received_done = True
                break
            try:
                obj = json.loads(data)
                choices = obj.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    chunks.append(content)
            except (json.JSONDecodeError, IndexError, KeyError):
                continue
    finally:
        resp.close()
    if not received_done:
        partial = "".join(chunks)
        raise RuntimeError(f"流式响应被截断（未收到 [DONE]），已接收 {len(partial)} 字符")
    return "".join(chunks)


def _retry_request(fn, *args, retries: int = 3, backoff: float = 2.0, **kwargs) -> str:
    """通用重试包装。"""
    last_err: Optional[Exception] = None
    for i in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = backoff * (2 ** i)
            log.warning(f"LLM 请求失败({i+1}/{retries}): {e}; {wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"LLM 请求重试 {retries} 次仍失败: {last_err}")


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
    if model:
        chosen.model_name = model
    if chosen.name == "aliyun":
        return AliyunClient(chosen)
    return OpenAICompatibleClient(chosen)


def safe_json_extract(text: str) -> Any:
    """从 LLM 输出中抽取首个 JSON 对象/数组，自动剥离 markdown 代码块，支持截断修复。"""
    text = text.strip()
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
                pass
        return _repair_truncated_json(text[s:], start, end)
    raise ValueError(f"无法从 LLM 输出中解析 JSON: {text[:200]}...")


def _repair_truncated_json(text: str, start: str, end: str) -> Any:
    """尝试修复被截断的 JSON：逐步去掉尾部不完整字段，补全括号。"""
    text = text.rstrip()

    # 策略：逐步缩短文本，每次尝试解析
    for trim in range(len(text)):
        candidate = text[: len(text) - trim] if trim > 0 else text
        if not candidate:
            break

        # 去掉尾部的逗号、冒号
        c = candidate.rstrip()
        while c and c[-1] in (",", ":"):
            c = c[:-1].rstrip()

        # 如果以孤立的 "key" 结尾（前面是逗号或 {），去掉整个 key
        if c.endswith('"'):
            key_start = c.rfind('"', 0, len(c) - 1)
            if key_start > 0:
                before = c[:key_start].rstrip()
                if before.endswith(",") or before.endswith("{"):
                    c = before.rstrip()
                    if c.endswith(","):
                        c = c[:-1].rstrip()

        # 补全缺失的括号
        stack = []
        for ch in c:
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("}", "]"):
                if stack and stack[-1] == ch:
                    stack.pop()
        repaired = c + "".join(reversed(stack))

        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            continue

    raise ValueError(f"JSON 修复失败，原始文本: {text[:200]}...")
