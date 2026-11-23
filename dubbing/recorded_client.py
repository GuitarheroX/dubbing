"""可留档、支持两种协议并验证思考开关的模型客户端。"""

from typing import Any

from .llm import APIClient, LLMError


class RecordedAPIClient(APIClient):
    """沿用现有接口、重试和采样配置，额外留存完整 JSON 响应供核对。"""

    def __init__(self, *args, protocol: str = "anthropic", thinking: str | None = None, **kwargs) -> None:
        if protocol not in ("anthropic", "openai"):
            raise ValueError(f"未知接口协议：{protocol}")
        self.protocol = protocol
        if thinking not in (None, "enabled", "disabled"):
            raise ValueError(f"未知思考设置：{thinking}")
        self.thinking = thinking
        super().__init__(*args, **kwargs)

    @property
    def url(self) -> str:
        if self.protocol == "openai":
            return f"{self.base_url}/chat/completions"
        return super().url

    def _headers(self) -> dict[str, str]:
        if self.protocol == "openai":
            if not self.api_key:
                raise LLMError("没有 API key：请设置 DEEPSEEK_API_KEY")
            return {"content-type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        return super()._headers()

    def _body(self, system: str, user: str, *, agent: str) -> dict[str, Any]:
        if self.protocol == "openai":
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature_for(agent),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if self.thinking is not None:
                body["enable_thinking"] = self.thinking == "enabled"
            return body
        body = super()._body(system, user, agent=agent)
        if self.thinking is not None:
            body["thinking"] = {"type": self.thinking}
        return body

    def complete(self, *, agent: str, system: str, user: str) -> str:
        self.last_response = None
        self.last_usage = None
        self.last_thinking_observation = None
        return super().complete(agent=agent, system=system, user=user)

    def _text_of(self, resp, *, agent: str) -> str:
        try:
            self.last_response = resp.json()
        except ValueError:
            pass
        data = self.last_response
        if self.protocol == "openai":
            if not isinstance(data, dict):
                raise LLMError(f"agent={agent} 返回的不是 JSON 对象")
            if "error" in data:
                raise LLMError(f"agent={agent} 调用失败：{str(data['error'])[:300]}")
            self.last_usage = data.get("usage")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LLMError(f"agent={agent} 的响应里没有 choices")
            choice = choices[0]
            message = choice.get("message")
            usage = self.last_usage or {}
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens")
            has_reasoning = isinstance(message, dict) and bool(
                message.get("reasoning_content") or message.get("reasoning")
            )
            self.last_thinking_observation = {
                "requested": self.thinking,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_content_present": has_reasoning,
            }
            if self.thinking == "disabled" and (
                has_reasoning or isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0
            ):
                raise LLMError("已请求 enable_thinking=false，但网关仍返回推理内容或 reasoning tokens；停止该任务")
            if choice.get("finish_reason") == "length":
                raise LLMError(f"agent={agent} 的输出被 max_tokens={self.max_tokens} 截断")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise LLMError(f"agent={agent} 的响应里没有正文，finish_reason={choice.get('finish_reason')}")
            return content.strip()
        if isinstance(data, dict):
            self.last_usage = data.get("usage")
            blocks = data.get("content")
            blocks = blocks if isinstance(blocks, list) else []
            has_reasoning = any(
                isinstance(block, dict)
                and (block.get("type") == "redacted_thinking" or block.get("type") == "thinking" and bool(block.get("thinking")))
                for block in blocks
            )
            usage = self.last_usage or {}
            reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            self.last_thinking_observation = {
                "requested": self.thinking,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_content_present": has_reasoning,
            }
            if self.thinking == "disabled" and (
                has_reasoning or isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0
            ):
                raise LLMError("已请求 thinking.type=disabled，但接口仍返回推理内容或 reasoning tokens；停止该任务")
        return super()._text_of(resp, agent=agent)

