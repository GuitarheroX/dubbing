"""文本嵌入客户端。

**建库与查询期必须用同一个模型、同一种归一化**，否则向量不可比。这里的约定是：
模型返回的向量一律由本模块做 L2 归一化，Chroma 用 cosine 空间，于是"余弦相似度"
与"内积"等价，服务端归一化与否都不影响结果。

默认走 GPU 服务器上 OpenAI 兼容的 ``/v1/embeddings``（TEI / vLLM 都提供）。
"""

from __future__ import annotations

import os
import time
from typing import Any, Protocol, Sequence

import requests

__all__ = [
    "DEFAULT_EMBED_MODEL",
    "DEFAULT_EMBED_URL",
    "EmbeddingClient",
    "EmbeddingError",
    "HttpEmbeddingClient",
    "LocalEmbeddingClient",
    "PendingEmbeddingClient",
    "build_embedding_client",
    "l2_normalize",
]

DEFAULT_EMBED_URL = "http://127.0.0.1:8080/v1/embeddings"
DEFAULT_EMBED_MODEL = "BAAI/bge-large-zh-v1.5"


class EmbeddingError(RuntimeError):
    """嵌入服务不可用、返回结构不对，或本机缺少本地推理依赖。"""


class EmbeddingClient(Protocol):
    """只要求两件事：叫什么、能不能把一批文本变成等长的向量。"""

    name: str
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def l2_normalize(vector: Sequence[float]) -> list[float]:
    norm = sum(float(x) * float(x) for x in vector) ** 0.5
    if norm <= 0:
        return [float(x) for x in vector]
    return [float(x) / norm for x in vector]


class PendingEmbeddingClient:
    """``--client pending`` 用：不发请求，真要用的时候报错。"""

    name = "pending"

    def __init__(self, model: str = DEFAULT_EMBED_MODEL) -> None:
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingError(
            "当前是 pending 客户端，不会调用嵌入服务。"
            "建库或跑 rag 管线时请设置 DUBBING_EMBED_URL（或 --embed-url），"
            "或在服务器上起 TEI。"
        )


class HttpEmbeddingClient:
    """OpenAI 兼容的 ``/v1/embeddings``。

    负责分批、重试与归一化；返回顺序与入参一致（按响应里的 ``index`` 排回来）。
    """

    def __init__(
        self,
        url: str = DEFAULT_EMBED_URL,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        api_key: str | None = None,
        batch_size: int = 128,
        timeout: float = 120.0,
        max_retries: int = 3,
        normalize: bool = True,
    ) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.normalize = normalize
        self.name = "http"

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post(self, texts: Sequence[str]) -> list[list[float]]:
        body = {"model": self.model, "input": list(texts), "encoding_format": "float"}
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.url, headers=self._headers(), json=body, timeout=self.timeout
                )
            except requests.RequestException as exc:  # 连不上 / 超时
                last = exc
            else:
                if resp.status_code < 400:
                    return self._parse(resp)
                last = EmbeddingError(f"HTTP {resp.status_code}：{resp.text[:300]}")
            if attempt < self.max_retries:
                time.sleep(min(2.0 * attempt, 8.0))
        raise EmbeddingError(f"嵌入服务重试 {self.max_retries} 次仍失败：{last}")

    def _parse(self, resp: requests.Response) -> list[list[float]]:
        try:
            payload: Any = resp.json()
        except ValueError as exc:
            raise EmbeddingError(f"嵌入服务返回的不是 JSON：{resp.text[:200]}") from exc
        if isinstance(payload, dict) and "error" in payload:
            raise EmbeddingError(f"嵌入服务报错：{str(payload['error'])[:300]}")
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise EmbeddingError(f"嵌入服务的响应里没有 data：{str(payload)[:300]}")
        ordered: list[tuple[int, list[float]]] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
                raise EmbeddingError(f"嵌入服务的第 {i} 条结果没有 embedding")
            index = row.get("index")
            ordered.append((index if isinstance(index, int) else i, [float(x) for x in row["embedding"]]))
        ordered.sort(key=lambda item: item[0])
        vectors = [v for _, v in ordered]
        return [l2_normalize(v) for v in vectors] if self.normalize else vectors

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        items = list(texts)
        if not items:
            return []
        out: list[list[float]] = []
        for start in range(0, len(items), self.batch_size):
            out.extend(self._post(items[start : start + self.batch_size]))
        return out


class LocalEmbeddingClient:
    """本机直接推理（``sentence-transformers``）。

    只有显式选择本地嵌入时才 import，避免把 torch 拖进基础依赖。
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确报错
            raise EmbeddingError(
                "本地嵌入需要 sentence-transformers：pip install sentence-transformers"
            ) from exc
        self.model = model
        self.normalize = normalize
        self.name = "local"
        self._batch_size = max(1, batch_size)
        self._encoder = SentenceTransformer(model, device=device)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._encoder.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in vectors]


def build_embedding_client(
    *,
    url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    local: bool = False,
    device: str | None = None,
    batch_size: int = 128,
    max_retries: int = 3,
) -> EmbeddingClient:
    """按参数与环境变量造一个嵌入客户端。

    环境变量：``DUBBING_EMBED_URL`` / ``DUBBING_EMBED_MODEL`` / ``DUBBING_EMBED_API_KEY`` /
    ``DUBBING_EMBED_LOCAL``（``1`` 表示本机推理）/ ``DUBBING_EMBED_DEVICE``。
    """
    resolved_model = model or os.environ.get("DUBBING_EMBED_MODEL") or DEFAULT_EMBED_MODEL
    use_local = local or (os.environ.get("DUBBING_EMBED_LOCAL") or "").strip() in {"1", "true", "yes"}
    if use_local:
        return LocalEmbeddingClient(
            resolved_model,
            device=device or os.environ.get("DUBBING_EMBED_DEVICE") or None,
            batch_size=batch_size,
        )
    return HttpEmbeddingClient(
        url or os.environ.get("DUBBING_EMBED_URL") or DEFAULT_EMBED_URL,
        resolved_model,
        api_key=api_key or os.environ.get("DUBBING_EMBED_API_KEY") or None,
        batch_size=batch_size,
        max_retries=max_retries,
    )
