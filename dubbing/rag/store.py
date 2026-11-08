"""本地 Chroma 向量库。

- ``PersistentClient`` + 本地目录（默认 ``data/poetry/chroma``），不启服务端；
- **``embedding_function=None``**：向量一律由调用方显式传入。Chroma 默认会挂一个英文
  MiniLM，静默改变向量空间，是最常见的翻车点；
- ``hnsw:space = cosine``，配合嵌入侧的 L2 归一化；
- 写入用 ``upsert``，同一个 ``chunk_id`` 重跑会覆盖而不是报错。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..models import PoetryHit

__all__ = [
    "MANIFEST_NAME",
    "ChromaStore",
    "StoreError",
    "read_manifest",
    "write_manifest",
]

MANIFEST_NAME = "manifest.json"
#: 一次 upsert 的条数上限（Chroma 默认最大 batch 约 5461，留足余量）。
UPSERT_BATCH = 1000


class StoreError(RuntimeError):
    """索引不存在、chromadb 缺失或用量不对。"""


def manifest_path(db_path: Path | str) -> Path:
    """索引的 manifest 放在 Chroma 目录的上一级。"""
    return Path(db_path).parent / MANIFEST_NAME


def read_manifest(db_path: Path | str) -> dict[str, Any]:
    path = manifest_path(db_path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def write_manifest(db_path: Path | str, payload: dict[str, Any]) -> Path:
    path = manifest_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


class ChromaStore:
    """一个 collection 的薄封装：写入、查询、计数。"""

    def __init__(
        self,
        path: Path | str,
        collection: str,
        *,
        create: bool = False,
        space: str = "cosine",
    ) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确报错
            raise StoreError("rag 管线需要 chromadb：pip install chromadb") from exc
        self.path = Path(path)
        self.collection_name = collection
        self.space = space
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._open(create=create)

    def _open(self, *, create: bool) -> Any:
        if create:
            return self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.space},
                embedding_function=None,
            )
        try:
            return self._client.get_collection(name=self.collection_name, embedding_function=None)
        except Exception as exc:  # chromadb 的异常类型随版本变化，统一包装
            raise StoreError(
                f"本地索引 {self.path} 里没有 collection {self.collection_name!r}；"
                "先用 scripts/build_poetry_index.py 建库"
            ) from exc

    # -- 写 --------------------------------------------------------------

    def upsert(
        self,
        *,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        total = len(ids)
        if not (total == len(documents) == len(metadatas) == len(embeddings)):
            raise StoreError("ids / documents / metadatas / embeddings 长度不一致")
        for start in range(0, total, UPSERT_BATCH):
            stop = start + UPSERT_BATCH
            self._collection.upsert(
                ids=list(ids[start:stop]),
                documents=list(documents[start:stop]),
                metadatas=[dict(m) for m in metadatas[start:stop]],
                embeddings=[list(v) for v in embeddings[start:stop]],
            )
        return total

    def reset(self) -> None:
        """删掉 collection 再建一个空的（``--reset`` 用）。"""
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.space},
            embedding_function=None,
        )

    # -- 读 --------------------------------------------------------------

    def count(self) -> int:
        return int(self._collection.count())

    def query(
        self,
        embedding: Sequence[float],
        k: int,
        *,
        exclude: set[str] | None = None,
        candidate_cap: int = 200,
    ) -> list[PoetryHit]:
        """取 top-k。

        ``exclude`` 里的 ``chunk_id`` 会被跳过（跨段去重用）；为了仍能凑够 k 条，
        会多取一些候选，最多 ``candidate_cap`` 条。
        """
        if k < 1:
            return []
        want = k if not exclude else min(candidate_cap, k + len(exclude) + k)
        result = self._collection.query(
            # 显式转成 Python float：传入 numpy 行时，list(row) 得到的是 np.float16 标量，
            # Chroma 会以"不认识的类型"拒掉
            query_embeddings=[[float(x) for x in embedding]],
            n_results=want,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[PoetryHit] = []
        for i, chunk_id in enumerate(ids):
            if exclude and chunk_id in exclude:
                continue
            meta = metadatas[i] if i < len(metadatas) and isinstance(metadatas[i], dict) else {}
            distance = float(distances[i]) if i < len(distances) else 0.0
            hits.append(
                PoetryHit(
                    chunk_id=str(chunk_id),
                    text=str(documents[i]) if i < len(documents) else "",
                    score=1.0 - distance,  # cosine 空间里 distance = 1 - 余弦
                    title=str(meta.get("title") or ""),
                    poet=str(meta.get("poet") or ""),
                    poem_id=str(meta.get("poem_id") or ""),
                    stanza_idx=meta.get("stanza_idx") if isinstance(meta.get("stanza_idx"), int) else None,
                )
            )
            if len(hits) >= k:
                break
        return hits

    @property
    def index_id(self) -> str:
        manifest = read_manifest(self.path)
        value = manifest.get("index_id")
        return str(value) if value else self.collection_name
