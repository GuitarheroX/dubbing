"""LLM 调用接口。

默认用 ``PendingClient``：不发任何请求，直接抛 ``NeedsLLM``，只把入参落盘。
真正跑模型时用 ``APIClient``，走 DeepSeek 的 Anthropic 兼容端点：

    POST https://api.deepseek.com/anthropic/v1/messages
    x-api-key: $DEEPSEEK_API_KEY

环境变量：

- ``DEEPSEEK_API_KEY``（也接受 ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``）
- ``DUBBING_MODEL``      默认 ``deepseek-flash``
- ``DUBBING_BASE_URL``   默认 ``https://api.deepseek.com/anthropic``
- ``DUBBING_TEMPERATURE`` 默认 0.7
- ``DUBBING_TEMPERATURE_A`` / ``DUBBING_TEMPERATURE_B`` 单独给某个 agent 设温度

``complete`` 返回**原始文本**，解析 JSON 由调用方负责（见 ``extract_json``），
这样模型偶尔多吐一句解释也不会把流程炸掉。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import requests

__all__ = [
    "load_dotenv",
    "LLMClient",
    "LLMError",
    "NeedsLLM",
    "PendingClient",
    "APIClient",
    "ScriptedClient",
    "extract_json",
    "build_client",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
]

DEFAULT_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_TEMPERATURE = 0.7
_KEY_ENVS = ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")

#: 这些状态码值得重试；4xx（除 408/429）说明请求本身有问题，重试无用。
_RETRYABLE = frozenset({408, 409, 429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """调用失败 / 返回无法解析。"""


class NeedsLLM(LLMError):
    """没有指定真实客户端：入参已落盘，等待人工或真实 API。"""


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, *, agent: str, system: str, user: str) -> str: ...


class PendingClient:
    """不发请求，只报出"请求已落盘"。"""

    name = "pending"

    def complete(self, *, agent: str, system: str, user: str) -> str:
        raise NeedsLLM(
            f"agent={agent!r} 需要一次模型调用，但当前是 pending 客户端。"
            " 用 --client api 走真实 API。"
        )


def load_dotenv(path: Path | None = None) -> list[str]:
    """把 .env 里的键值补进 ``os.environ``，**不覆盖**已存在的变量。

    只做最简单的 ``KEY=VALUE`` 解析：忽略空行和 ``#`` 注释，去掉成对引号。
    返回实际补进去的变量名，方便命令行提示一句"从 .env 载入了什么"。
    """
    candidates = (
        [Path(path)]
        if path is not None
        else [Path(".env"), Path(__file__).resolve().parent.parent / ".env"]
    )
    loaded: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            if name.startswith("export "):
                name = name[len("export ") :].strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if name and name not in os.environ:
                os.environ[name] = value
                loaded.append(name)
    return loaded


def _api_key_from_env() -> str | None:
    for name in _KEY_ENVS:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


class APIClient:
    """DeepSeek 的 Anthropic 兼容端点。

    只依赖 ``requests``。重试只针对 429/5xx 和网络错误；4xx 直接报错，
    因为那说明请求本身有问题，重试只会浪费钱。
    """

    name = "api"

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8000,
        temperature: float | None = None,
        temperature_by_agent: dict[str, float] | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.environ.get("DUBBING_MODEL") or DEFAULT_MODEL
        self.base_url = (
            base_url or os.environ.get("DUBBING_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key or _api_key_from_env()
        self.max_tokens = max_tokens
        if temperature is None:
            temperature = float(os.environ.get("DUBBING_TEMPERATURE") or DEFAULT_TEMPERATURE)
        self.temperature = temperature

        per_agent = dict(temperature_by_agent or {})
        for agent, var in (("a", "DUBBING_TEMPERATURE_A"), ("b", "DUBBING_TEMPERATURE_B")):
            if agent not in per_agent and os.environ.get(var):
                per_agent[agent] = float(os.environ[var])
        #: 按 agent 覆盖温度：A 要发散，B 要严谨，两者不该用同一个值
        self.temperature_by_agent = per_agent
        self.timeout = timeout
        self.max_retries = max_retries
        #: 最近一次调用的用量与温度，给编排器记进 meta.json
        self.last_usage: dict[str, Any] | None = None
        self.last_temperature: float | None = None

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/messages"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMError(
                "没有 API key：请设置环境变量 "
                + " 或 ".join(_KEY_ENVS)
                + "（例如 export DEEPSEEK_API_KEY=sk-...）"
            )
        return {
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

    def temperature_for(self, agent: str) -> float:
        return self.temperature_by_agent.get(agent, self.temperature)

    def _body(self, system: str, user: str, *, agent: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature_for(agent),
            "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
        }

    def complete(self, *, agent: str, system: str, user: str) -> str:
        headers = self._headers()
        self.last_temperature = self.temperature_for(agent)
        body = self._body(system, user, agent=agent)

        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.url, json=body, headers=headers, timeout=(10, self.timeout)
                )
            except requests.RequestException as exc:
                last_error = f"网络错误：{exc}"
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                raise LLMError(f"agent={agent} 调用失败：{last_error}") from exc

            if resp.status_code in _RETRYABLE and attempt < self.max_retries:
                wait = min(2 ** (attempt - 1), 8)
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    wait = min(int(retry_after), 30)
                last_error = f"HTTP {resp.status_code}：{self._snippet(resp)}"
                time.sleep(wait)
                continue

            if resp.status_code >= 400:
                raise LLMError(
                    f"agent={agent} 调用失败：HTTP {resp.status_code}：{self._snippet(resp)}"
                )

            return self._text_of(resp, agent=agent)

        raise LLMError(f"agent={agent} 重试 {self.max_retries} 次仍失败：{last_error}")

    # -- 响应处理 --------------------------------------------------------

    @staticmethod
    def _snippet(resp: requests.Response) -> str:
        raw = getattr(resp, "text", "") or ""
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return raw[:500] or "<空响应体>"
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict) and err.get("message"):
                return str(err["message"])[:500]
            if isinstance(err, str):
                return err[:500]
        return raw[:500]

    def _text_of(self, resp: requests.Response, *, agent: str) -> str:
        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMError(f"agent={agent} 返回的不是 JSON：{self._snippet(resp)}") from exc

        if isinstance(data, dict) and "error" in data:
            raise LLMError(f"agent={agent} 调用失败：{data['error']}")

        self.last_usage = data.get("usage") if isinstance(data, dict) else None

        blocks = data.get("content") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise LLMError(f"agent={agent} 的响应里没有 content：{str(data)[:300]}")

        # 只取 text 块；thinking 块要跳过，否则会把推理过程当成正文
        text = "".join(
            b.get("text", "")
            for b in blocks
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        if not text:
            raise LLMError(f"agent={agent} 的响应里没有文本块：{str(data)[:300]}")

        if data.get("stop_reason") == "max_tokens":
            raise LLMError(
                f"agent={agent} 的输出被 max_tokens={self.max_tokens} 截断，"
                "JSON 必然不完整；调大 max_tokens 或把 section 拆小。"
            )
        return text


class ScriptedClient:
    """测试用：按调用顺序吐出预设文本，并记录每次入参。"""

    name = "scripted"

    def __init__(self, script: dict[str, list[str]] | list[str]) -> None:
        self._by_agent: dict[str, list[str]] | None = (
            {k: list(v) for k, v in script.items()} if isinstance(script, dict) else None
        )
        self._queue: list[str] = [] if self._by_agent is not None else list(script)  # type: ignore[arg-type]
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, agent: str, system: str, user: str) -> str:
        self.calls.append({"agent": agent, "system": system, "user": user})
        if self._by_agent is not None:
            queue = self._by_agent.get(agent)
            if not queue:
                raise LLMError(f"脚本里没有给 agent={agent!r} 预留响应")
            return queue.pop(0)
        if not self._queue:
            raise LLMError("脚本已用尽")
        return self._queue.pop(0)


def extract_json(text: str) -> Any:
    """从模型输出里取出 JSON。容忍 ```json 围栏和前后废话。"""
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", stripped, re.S)
    candidates = [stripped]
    if fence:
        candidates.insert(0, fence.group(1).strip())

    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"模型输出不是合法 JSON：{exc}") from exc
    raise LLMError(f"模型输出里找不到 JSON：{stripped[:200]!r}")


def build_client(kind: str | None = None, **kwargs: Any) -> LLMClient:
    kind = kind or os.environ.get("DUBBING_CLIENT") or "pending"
    if kind == "pending":
        return PendingClient()
    if kind == "api":
        return APIClient(**kwargs)
    raise KeyError(f"未知客户端 {kind!r}（可选 pending / api）")
