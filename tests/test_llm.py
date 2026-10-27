"""API 客户端：只测请求怎么拼、响应怎么解析，不联网。"""

import json

import pytest

from dubbing import llm as llm_mod
from dubbing.llm import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    APIClient,
    LLMError,
    NeedsLLM,
    PendingClient,
    build_client,
    extract_json,
)

KEYS = ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        if text is not None:
            self.text = text
        else:
            self.text = json.dumps(payload, ensure_ascii=False) if payload is not None else ""
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _ok(text, *, stop_reason="end_turn", usage=None):
    return FakeResponse(
        payload={
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "usage": usage or {"input_tokens": 11, "output_tokens": 22},
        }
    )


@pytest.fixture
def key(monkeypatch):
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.delenv("DUBBING_MODEL", raising=False)
    monkeypatch.delenv("DUBBING_BASE_URL", raising=False)
    return "sk-test"


def test_posts_to_deepseek_anthropic_endpoint(monkeypatch, key):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, body=json, headers=headers, timeout=timeout)
        return _ok("你好")

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)

    client = APIClient()
    assert client.complete(agent="b", system="SYS", user="USR") == "你好"

    assert seen["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["model"] == DEFAULT_MODEL
    assert seen["body"]["system"] == "SYS"
    assert seen["body"]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "USR"}]}
    ]
    assert client.last_usage == {"input_tokens": 11, "output_tokens": 22}


def test_ignores_thinking_blocks(monkeypatch, key):
    monkeypatch.setattr(
        llm_mod.requests,
        "post",
        lambda *a, **k: FakeResponse(
            payload={
                "content": [
                    {"type": "thinking", "thinking": "内部推理，不该出现在结果里"},
                    {"type": "text", "text": '{"ok": true}'},
                ],
                "stop_reason": "end_turn",
            }
        ),
    )
    assert APIClient().complete(agent="a", system="s", user="u") == '{"ok": true}'


def test_model_and_base_url_from_env(monkeypatch, key):
    monkeypatch.setenv("DUBBING_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("DUBBING_BASE_URL", "https://example.test/anthropic/")
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, model=json["model"])
        return _ok("x")

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    APIClient().complete(agent="a", system="s", user="u")

    assert seen["model"] == "deepseek-v4-pro"
    assert seen["url"] == "https://example.test/anthropic/v1/messages"
    assert DEFAULT_BASE_URL.endswith("/anthropic")


def test_missing_key_raises_with_hint(monkeypatch):
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(LLMError, match="API key"):
        APIClient().complete(agent="a", system="s", user="u")


def test_4xx_is_not_retried(monkeypatch, key):
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return FakeResponse(status_code=400, text="Model Not Exist")

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)

    with pytest.raises(LLMError, match="HTTP 400"):
        APIClient().complete(agent="a", system="s", user="u")
    assert len(calls) == 1


def test_5xx_is_retried(monkeypatch, key):
    responses = [FakeResponse(status_code=503, text="busy"), _ok("终于好了")]
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)

    assert APIClient().complete(agent="a", system="s", user="u") == "终于好了"


def test_network_error_is_retried_then_raised(monkeypatch, key):
    import requests

    monkeypatch.setattr(
        llm_mod.requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("boom"))
    )
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)

    with pytest.raises(LLMError, match="网络错误"):
        APIClient(max_retries=2).complete(agent="a", system="s", user="u")


def test_truncated_output_is_reported(monkeypatch, key):
    monkeypatch.setattr(llm_mod.requests, "post", lambda *a, **k: _ok('{"section": "C1"', stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens"):
        APIClient().complete(agent="b", system="s", user="u")


def test_empty_content_is_reported(monkeypatch, key):
    monkeypatch.setattr(
        llm_mod.requests,
        "post",
        lambda *a, **k: FakeResponse(payload={"content": [], "stop_reason": "end_turn"}),
    )
    with pytest.raises(LLMError, match="没有文本块"):
        APIClient().complete(agent="b", system="s", user="u")


def test_error_payload_is_reported(monkeypatch, key):
    monkeypatch.setattr(
        llm_mod.requests,
        "post",
        lambda *a, **k: FakeResponse(payload={"error": {"message": "quota exceeded"}}),
    )
    with pytest.raises(LLMError, match="quota exceeded"):
        APIClient().complete(agent="b", system="s", user="u")


# -- extract_json --------------------------------------------------------


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('好的：\n```json\n{"a": 2}\n```\n以上') == {"a": 2}


def test_extract_json_with_prose():
    assert extract_json('这是结果 {"a": 3} 希望有帮助') == {"a": 3}


def test_extract_json_gives_up_clearly():
    with pytest.raises(LLMError, match="找不到 JSON"):
        extract_json("完全不是 JSON")


# -- 客户端选择 ----------------------------------------------------------


def test_pending_client_points_at_the_flag():
    with pytest.raises(NeedsLLM, match="--client api"):
        PendingClient().complete(agent="a", system="s", user="u")


def test_build_client_defaults_to_pending(monkeypatch):
    monkeypatch.delenv("DUBBING_CLIENT", raising=False)
    assert build_client().name == "pending"


def test_build_client_env_switch(monkeypatch, key):
    monkeypatch.setenv("DUBBING_CLIENT", "api")
    client = build_client()
    assert client.name == "api"
    assert client.model == DEFAULT_MODEL


# -- .env ----------------------------------------------------------------


def test_load_dotenv_fills_missing_only(monkeypatch, tmp_path):
    from dubbing.llm import load_dotenv

    for name in KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    env = tmp_path / ".env"
    env.write_text(
        "# 注释\nexport DEEPSEEK_API_KEY=sk-from-file\nANTHROPIC_API_KEY=sk-should-not-win\n\n",
        encoding="utf-8",
    )
    loaded = load_dotenv(env)

    import os

    assert loaded == ["DEEPSEEK_API_KEY"]
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-from-file"
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_load_dotenv_strips_quotes(monkeypatch, tmp_path):
    import os

    from dubbing.llm import load_dotenv

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('DEEPSEEK_API_KEY="sk-quoted"\n', encoding="utf-8")
    load_dotenv(env)
    assert os.environ["DEEPSEEK_API_KEY"] == "sk-quoted"


def test_load_dotenv_missing_file_is_noop(tmp_path):
    from dubbing.llm import load_dotenv

    assert load_dotenv(tmp_path / "nope") == []


# -- 温度 ----------------------------------------------------------------


def _capture_body(monkeypatch):
    seen = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update(json)
        return _ok("x")

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    return seen


def test_default_temperature(monkeypatch, key):
    monkeypatch.delenv("DUBBING_TEMPERATURE", raising=False)
    seen = _capture_body(monkeypatch)
    APIClient().complete(agent="b", system="s", user="u")
    assert seen["temperature"] == 0.7


def test_temperature_from_env(monkeypatch, key):
    monkeypatch.setenv("DUBBING_TEMPERATURE", "1.1")
    seen = _capture_body(monkeypatch)
    APIClient().complete(agent="b", system="s", user="u")
    assert seen["temperature"] == 1.1


def test_temperature_can_differ_per_agent(monkeypatch, key):
    monkeypatch.delenv("DUBBING_TEMPERATURE", raising=False)
    monkeypatch.delenv("DUBBING_TEMPERATURE_A", raising=False)
    monkeypatch.delenv("DUBBING_TEMPERATURE_B", raising=False)
    seen = _capture_body(monkeypatch)
    client = APIClient(temperature_by_agent={"a": 1.2, "b": 0.4})

    client.complete(agent="a", system="s", user="u")
    assert seen["temperature"] == 1.2
    assert client.last_temperature == 1.2

    client.complete(agent="b", system="s", user="u")
    assert seen["temperature"] == 0.4


def test_per_agent_temperature_from_env(monkeypatch, key):
    monkeypatch.delenv("DUBBING_TEMPERATURE_A", raising=False)
    monkeypatch.setenv("DUBBING_TEMPERATURE_B", "0.3")
    seen = _capture_body(monkeypatch)
    client = APIClient()

    client.complete(agent="b", system="s", user="u")
    assert seen["temperature"] == 0.3

    client.complete(agent="a", system="s", user="u")
    assert seen["temperature"] == 0.7  # 没设的用通用值
