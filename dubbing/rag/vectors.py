"""向量文件（``vectors.npy`` + ``vectors.json``）。

两台机器的分工靠它对齐：**服务器**用 GPU 把 ``chunks.jsonl`` 编码成向量，**本机**把向量灌进
Chroma，全程不需要本机有 GPU，也不需要起嵌入服务。

配对的两份东西：

- ``vectors.npy``：float16/float32 的 ``(N, dim)`` 矩阵，行序与 ``chunks.jsonl`` 一致；
- ``vectors.json``：sidecar，记录模型、维度、条数、dtype、归一化，以及 **chunks 文件的 sha1**。

sidecar 里的 sha1 是为了防一个很容易犯的错：换过切块参数之后，向量还是旧的一份。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "SIDECAR_SUFFIX",
    "VectorFileError",
    "VectorMeta",
    "file_sha1",
    "sidecar_path",
    "save_vectors",
    "load_vectors",
]


class VectorFileError(RuntimeError):
    """向量文件缺失、形状不对，或和 chunks 文件对不上。"""


SIDECAR_SUFFIX = ".json"


def sidecar_path(vectors_path: Path | str) -> Path:
    """``vectors.npy`` → ``vectors.json``。"""
    path = Path(vectors_path)
    return path.with_suffix(SIDECAR_SUFFIX)


def file_sha1(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def count_lines(path: Path | str) -> int:
    with Path(path).open("r", encoding="utf-8") as fh:
        return sum(1 for _ in fh)


@dataclass(frozen=True)
class VectorMeta:
    """``vectors.json`` 的内容。"""

    model: str
    dim: int
    count: int
    dtype: str = "float16"
    normalize: bool = True
    pooling: str = "cls"
    max_length: int | None = None
    device: str = ""
    chunks_file: str = ""
    chunks_sha1: str = ""
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "dim": self.dim,
            "count": self.count,
            "dtype": self.dtype,
            "normalize": self.normalize,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "device": self.device,
            "chunks": {"file": self.chunks_file, "sha1": self.chunks_sha1},
            "created_at": self.created_at
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_json(cls, obj: Any) -> "VectorMeta":
        if not isinstance(obj, dict):
            raise VectorFileError("vectors.json 不是一个 JSON 对象")
        chunks = obj.get("chunks") if isinstance(obj.get("chunks"), dict) else {}
        known = {
            "model",
            "dim",
            "count",
            "dtype",
            "normalize",
            "pooling",
            "max_length",
            "device",
            "chunks",
            "created_at",
        }
        return cls(
            model=str(obj.get("model") or ""),
            dim=int(obj.get("dim") or 0),
            count=int(obj.get("count") or 0),
            dtype=str(obj.get("dtype") or "float16"),
            normalize=bool(obj.get("normalize", True)),
            pooling=str(obj.get("pooling") or "cls"),
            max_length=obj.get("max_length") if isinstance(obj.get("max_length"), int) else None,
            device=str(obj.get("device") or ""),
            chunks_file=str(chunks.get("file") or ""),
            chunks_sha1=str(chunks.get("sha1") or ""),
            created_at=str(obj.get("created_at") or ""),
            extra={k: v for k, v in obj.items() if k not in known},
        )


def save_vectors(
    vectors_path: Path | str,
    vectors: Sequence[Sequence[float]],
    meta: VectorMeta,
) -> Path:
    """写 ``vectors.npy`` 与 sidecar，并核对 meta.count 与实际条数。"""
    import numpy as np  # rag 资产才需要 numpy

    array = np.asarray(vectors, dtype=meta.dtype)
    if array.ndim != 2:
        raise VectorFileError(f"向量应当是二维矩阵，得到 shape={array.shape}")
    if meta.dim and array.shape[1] != meta.dim:
        raise VectorFileError(f"向量维度 {array.shape[1]} 与 meta.dim={meta.dim} 不符")
    if meta.count and array.shape[0] != meta.count:
        raise VectorFileError(f"向量条数 {array.shape[0]} 与 meta.count={meta.count} 不符")

    path = Path(vectors_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    sidecar = sidecar_path(path)
    sidecar.write_text(
        json.dumps(meta.to_json(), ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return sidecar


def load_vectors(
    vectors_path: Path | str,
    *,
    expected_count: int | None = None,
    chunks_file: Path | str | None = None,
    chunks_sha1: str | None = None,
    mmap: bool = False,
) -> tuple[Any, VectorMeta]:
    """读向量与 sidecar，并在能校验的地方当场校验。

    校验分三层，任何一层不对都直接报错——向量和切块错位是最难查的一类问题：

    1. 行数 == sidecar.count（== ``expected_count``，如果给了）；
    2. ``meta.dim`` 非零时，列数与它一致；
    3. 给了 ``chunks_sha1`` 时，与 sidecar 里记录的 sha1 一致（可选，慢一点但最稳）。

    ``mmap=True`` 时不把整个矩阵读进内存（灌库用，26 万条也只有几十 MB 常驻）。
    """
    import numpy as np

    path = Path(vectors_path)
    if not path.is_file():
        raise VectorFileError(f"找不到向量文件 {path}")
    sidecar = sidecar_path(path)
    if not sidecar.is_file():
        raise VectorFileError(f"找不到向量文件的说明 {sidecar}（应当与 vectors.npy 一起产出）")

    try:
        meta = VectorMeta.from_json(json.loads(sidecar.read_text(encoding="utf-8")))
    except ValueError as exc:
        raise VectorFileError(f"{sidecar} 不是合法 JSON：{exc}") from exc

    vectors = np.load(path, mmap_mode="r" if mmap else None)
    if vectors.ndim != 2:
        raise VectorFileError(f"{path} 应当是二维矩阵，得到 shape={vectors.shape}")
    if meta.dim and vectors.shape[1] != meta.dim:
        raise VectorFileError(
            f"{path} 的向量维度是 {vectors.shape[1]}，但 {sidecar.name} 里写的是 {meta.dim}"
        )
    if meta.count and vectors.shape[0] != meta.count:
        raise VectorFileError(
            f"{path} 有 {vectors.shape[0]} 条向量，但 {sidecar.name} 里写的是 {meta.count}"
        )
    if expected_count is not None and vectors.shape[0] != expected_count:
        raise VectorFileError(
            f"向量条数 {vectors.shape[0]} 与 chunks 文件的行数 {expected_count} 不一致"
            "——多半是切块参数变了但向量没重跑"
        )
    if not meta.dim:
        meta = replace(meta, dim=int(vectors.shape[1]))

    if chunks_file is not None and chunks_sha1 and meta.chunks_sha1:
        if meta.chunks_sha1 != chunks_sha1:
            raise VectorFileError(
                f"chunks 文件与向量对不上：\n"
                f"  本地 {Path(chunks_file).name} 的 sha1 = {chunks_sha1}\n"
                f"  {sidecar.name} 记录的 sha1 = {meta.chunks_sha1}\n"
                "要么重跑 embed_poetry.py，要么删掉 chunks.jsonl 重新切块。"
            )
    return vectors, meta


def iter_batches(items: Sequence[Any], size: int) -> Iterable[tuple[int, Sequence[Any]]]:
    for start in range(0, len(items), size):
        yield start, items[start : start + size]
