"""检索器：把一段中文译文变成几条参考诗句。

管线里只有这一处会碰嵌入服务和向量库，所以它也是唯一需要"注入替身"的地方
（测试用 ``FakeRetriever`` 即可，不必装 chromadb）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..models import PoetryHit
from .embedding import EmbeddingClient, EmbeddingError, build_embedding_client
from .store import ChromaStore

__all__ = ["DEFAULT_COLLECTION", "DEFAULT_DB", "DEFAULT_K", "Retriever", "build_retriever"]

DEFAULT_DB = Path("data/poetry/chroma")
DEFAULT_COLLECTION = "poetry_stanza256_v1"
#: 默认取 4 条：一个诗节约 40 token，4 条约 160 token。
DEFAULT_K = 4


@dataclass
class Retriever:
    """嵌入客户端 + 本地向量库；``k`` 是默认条数。"""

    embedding: EmbeddingClient
    store: ChromaStore
    k: int = DEFAULT_K

    @property
    def collection(self) -> str:
        return self.store.collection_name

    @property
    def index_id(self) -> str:
        return self.store.index_id

    def search(
        self,
        query_text: str,
        *,
        k: int | None = None,
        exclude: set[str] | None = None,
    ) -> list[PoetryHit]:
        text = (query_text or "").strip()
        if not text:
            return []
        vectors = self.embedding.embed([text])
        if not vectors:
            raise EmbeddingError("嵌入服务没有返回查询向量")
        return self.store.query(vectors[0], k or self.k, exclude=exclude)


def build_retriever(
    *,
    db: Path | str | None = None,
    collection: str | None = None,
    k: int | None = None,
    embed_url: str | None = None,
    embed_model: str | None = None,
    api_key: str | None = None,
    local: bool = False,
    device: str | None = None,
) -> Retriever:
    """按参数与环境变量组装检索器。

    环境变量：``DUBBING_RAG_DB`` / ``DUBBING_RAG_COLLECTION`` / ``DUBBING_RAG_K``，
    嵌入相关见 :func:`dubbing.rag.embedding.build_embedding_client`。
    """
    resolved_k = k if k is not None else int(os.environ.get("DUBBING_RAG_K") or DEFAULT_K)
    store = ChromaStore(
        db or os.environ.get("DUBBING_RAG_DB") or DEFAULT_DB,
        collection or os.environ.get("DUBBING_RAG_COLLECTION") or DEFAULT_COLLECTION,
    )
    embedding = build_embedding_client(
        url=embed_url, model=embed_model, api_key=api_key, local=local, device=device
    )
    return Retriever(embedding=embedding, store=store, k=resolved_k)
