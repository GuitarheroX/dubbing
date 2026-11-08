"""嵌入客户端：分批、排序、归一化、重试——都不联网。"""

import json

import pytest

from dubbing.rag import embedding as embedding_mod
from dubbing.rag.embedding import (
    DEFAULT_EMBED_MODEL,
    EmbeddingError,
    HttpEmbeddingClient,
    PendingEmbeddingClient,
    build_embedding_client,
    l2_normalize,
)


class FakeResponse:
    def __init__(self, payload, status: int = 200):
        self._payload = payload
        self.status_code = status
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload


def _reply(vectors, *, reverse: bool = False):
    """按 OpenAI 的形状造一条响应；``reverse`` 用来验证按 index 排回来。"""
    rows = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if reverse:
        rows = list(reversed(rows))
    return FakeResponse({"data": rows, "model": DEFAULT_EMBED_MODEL})


def _client(monkeypatch, responses, **kwargs):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers})
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(embedding_mod.requests, "post", fake_post)
    return HttpEmbeddingClient(url="http://embed.test/v1/embeddings", **kwargs), calls


def test_l2_normalize_makes_unit_vectors():
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]  # 全零不炸


def test_embed_sends_batches_and_keeps_order(monkeypatch):
    client, calls = _client(monkeypatch, [_reply([[1.0, 0.0], [0.0, 2.0]], reverse=True)], batch_size=2)
    vectors = client.embed(["甲", "乙"])

    assert len(calls) == 1
    assert calls[0]["json"] == {
        "model": DEFAULT_EMBED_MODEL,
        "input": ["甲", "乙"],
        "encoding_format": "float",
    }
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]  # 排回原顺序并归一化


def test_embed_splits_into_batches(monkeypatch):
    client, calls = _client(
        monkeypatch,
        [_reply([[1.0, 0.0]]), _reply([[0.0, 1.0]])],
        batch_size=1,
    )
    assert client.embed(["甲", "乙"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert [c["json"]["input"] for c in calls] == [["甲"], ["乙"]]


def test_embed_empty_input_does_not_call_the_service(monkeypatch):
    client, calls = _client(monkeypatch, [])
    assert client.embed([]) == []
    assert calls == []


def test_embed_retries_then_succeeds(monkeypatch):
    client, calls = _client(
        monkeypatch,
        [FakeResponse({"error": "busy"}, status=503), _reply([[1.0, 0.0]])],
        max_retries=2,
    )
    monkeypatch.setattr(embedding_mod.time, "sleep", lambda _s: None)
    assert client.embed(["甲"]) == [[1.0, 0.0]]
    assert len(calls) == 2


def test_embed_gives_up_after_max_retries(monkeypatch):
    client, _ = _client(
        monkeypatch,
        [FakeResponse("boom", status=500) for _ in range(3)],
        max_retries=3,
    )
    monkeypatch.setattr(embedding_mod.time, "sleep", lambda _s: None)
    with pytest.raises(EmbeddingError, match="重试 3 次仍失败"):
        client.embed(["甲"])


def test_embed_rejects_a_response_without_data(monkeypatch):
    client, _ = _client(monkeypatch, [_reply([])])
    with pytest.raises(EmbeddingError, match="没有 data"):
        client.embed(["甲"])


def test_embed_rejects_a_row_without_embedding(monkeypatch):
    client, _ = _client(monkeypatch, [FakeResponse({"data": [{"index": 0}]})])
    with pytest.raises(EmbeddingError, match="没有 embedding"):
        client.embed(["甲"])


def test_embed_rejects_non_json(monkeypatch):
    client, _ = _client(monkeypatch, [FakeResponse("not json at all")])
    with pytest.raises(EmbeddingError, match="不是 JSON"):
        client.embed(["甲"])


def test_headers_carry_the_api_key(monkeypatch):
    client, calls = _client(monkeypatch, [_reply([[1.0]])], api_key="sk-x")
    client.embed(["甲"])
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-x"


def test_pending_client_explains_itself():
    with pytest.raises(EmbeddingError, match="pending"):
        PendingEmbeddingClient().embed(["甲"])


def test_build_client_prefers_arguments_over_env(monkeypatch):
    monkeypatch.setenv("DUBBING_EMBED_URL", "http://from-env/v1/embeddings")
    monkeypatch.setenv("DUBBING_EMBED_MODEL", "env-model")
    default = build_embedding_client()
    assert default.url == "http://from-env/v1/embeddings"
    assert default.model == "env-model"

    explicit = build_embedding_client(url="http://x/v1/embeddings", model="m")
    assert (explicit.url, explicit.model) == ("http://x/v1/embeddings", "m")


def test_build_client_defaults_to_localhost_tei(monkeypatch):
    client = build_embedding_client()
    assert client.url == "http://127.0.0.1:8080/v1/embeddings"
    assert client.model == DEFAULT_EMBED_MODEL
