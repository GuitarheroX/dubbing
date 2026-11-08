#!/usr/bin/env python3
"""在**服务器**上把诗句编码成向量（这一步需要 GPU，也只在这一步需要）。

它读本机切好的 ``chunks.jsonl``，产出 ``vectors.npy`` + ``vectors.json``；把这两个文件拷回本机，
再用 ``scripts/build_poetry_index.py --vectors`` 灌进本地 Chroma。整条链路里本机不需要 GPU，
也不需要起任何嵌入服务。

服务器上准备：

    pip install sentence-transformers          # 会自动带上 torch
    # 有网就让它自己下模型；没网就先把模型目录拷过去，用 --model /path/to/bge-large-zh-v1.5

先跑一小段验证（强烈建议）：

    python scripts/embed_poetry.py --chunks chunks.jsonl --limit 2000 \\
        --out vectors.head.npy --device cuda

全量：

    python scripts/embed_poetry.py --chunks chunks.jsonl --out vectors.npy \\
        --model BAAI/bge-large-zh-v1.5 --device cuda --batch 64 --dtype float16

``--dtype float16`` 让文件小一半（26 万条 × 1024 维：fp32 约 1.07 GB，fp16 约 535 MB）；
Chroma 用 cosine 距离，排序不受 dtype 影响。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.rag.vectors import VectorFileError, VectorMeta, save_vectors, sidecar_path  # noqa: E402

__all__ = ["main", "build_encoder", "embed_chunks", "count_and_hash"]

PROGRESS_EVERY = 10000
Encoder = Callable[[Sequence[str]], "list[list[float]]"]


def count_and_hash(path: Path | str, *, chunk_size: int = 1 << 20) -> tuple[int, str]:
    """一趟读完：数行数 + 算 sha1。"""
    import hashlib

    digest = hashlib.sha1()
    lines = 0
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            digest.update(block)
            lines += block.count(b"\n")
    return lines, digest.hexdigest()


def build_encoder(
    model: str,
    *,
    device: str | None = None,
    batch_size: int = 64,
    max_length: int | None = None,
) -> Encoder:
    """把 ``sentence-transformers`` 包成一个"一批文本 → 一批向量"的函数。"""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - 依赖缺失时的明确报错
        raise SystemExit(
            "服务器上需要 sentence-transformers：pip install sentence-transformers"
        ) from exc

    encoder = SentenceTransformer(model, device=device)
    if max_length:
        encoder.max_seq_length = max_length

    def encode(texts: Sequence[str]) -> list[list[float]]:
        vectors = encoder.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors

    return encode


def _read_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_subset(source: Path, rows: list[dict], limit: int) -> Path:
    """``--limit`` 时把用到的那些行单独存一份，让 chunks 与 vectors 仍然一一对应。"""
    target = source.with_name(f"{source.stem}.limit{limit}{source.suffix}")
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target


def embed_chunks(
    rows: Sequence[dict],
    encode: Encoder,
    *,
    dtype: str = "float16",
    batch_size: int = 64,
    quiet: bool = False,
) -> "object":
    """分批编码，边编边填进最终矩阵（避免把 fp32 全量留在内存里）。"""
    import numpy as np

    total = len(rows)
    out = None
    done = 0
    started = time.time()
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        vectors = np.asarray(encode([row.get("text", "") for row in batch]))
        if vectors.ndim != 2:
            raise VectorFileError(f"编码器返回的不是二维矩阵：shape={vectors.shape}")
        if out is None:
            out = np.empty((total, vectors.shape[1]), dtype=dtype)
        out[start : start + vectors.shape[0]] = vectors
        done += vectors.shape[0]
        if not quiet and (done >= PROGRESS_EVERY or done == total):
            rate = done / max(time.time() - started, 1e-6)
            print(f"  已编码 {done}/{total}（{rate:.0f} 条/秒）", file=sys.stderr, flush=True)
    if out is None:
        out = np.empty((0, 0), dtype=dtype)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 GPU 把诗句编码成向量文件（在服务器上跑）")
    parser.add_argument("--chunks", default=str(REPO_ROOT / "data" / "poetry" / "chunks.jsonl"))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "data" / "poetry" / "vectors.npy"), help="输出 .npy"
    )
    parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5", help="HF 模型名或本地模型目录")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu，默认自动挑")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--max-length", type=int, default=None, help="默认用模型自带（bge 是 512）")
    parser.add_argument("--limit", type=int, default=None, help="只编码前 N 条（跑通流程用）")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch < 1 or (args.limit is not None and args.limit < 1):
        print("--batch 与 --limit 必须是正整数", file=sys.stderr)
        return 2

    chunks = Path(args.chunks)
    if not chunks.is_file():
        print(f"找不到切块文件 {chunks}；先在本机跑 build_poetry_index.py --dry-run", file=sys.stderr)
        return 1

    started = time.time()
    total_lines, _ = count_and_hash(chunks)
    rows = _read_rows(chunks, args.limit)
    if args.limit is not None and args.limit < total_lines:
        chunks = _write_subset(chunks, rows, args.limit)
        print(f"--limit {args.limit}：只编码前 {args.limit} 条，另存 {chunks}", file=sys.stderr)
    if not rows:
        print(f"{chunks} 里没有可编码的行", file=sys.stderr)
        return 1

    count, sha1 = count_and_hash(chunks)
    if count != len(rows):
        print(f"读到的行数 {len(rows)} 与文件行数 {count} 不一致", file=sys.stderr)
        return 1

    print(f"编码 {count} 条，模型 {args.model}，批次 {args.batch}，dtype {args.dtype}", file=sys.stderr)
    encode = build_encoder(
        args.model, device=args.device, batch_size=args.batch, max_length=args.max_length
    )
    vectors = embed_chunks(rows, encode, dtype=args.dtype, batch_size=args.batch, quiet=args.quiet)

    meta = VectorMeta(
        model=args.model,
        dim=int(vectors.shape[1]),
        count=int(vectors.shape[0]),
        dtype=args.dtype,
        normalize=True,
        pooling="cls",
        max_length=args.max_length or 512,
        device=args.device or "",
        chunks_file=str(chunks),
        chunks_sha1=sha1,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    sidecar = save_vectors(args.out, vectors, meta)

    size_mb = Path(args.out).stat().st_size / 1e6
    print(
        f"\n完成：{vectors.shape[0]} 条 × {vectors.shape[1]} 维 → {args.out}（{size_mb:.0f} MB）\n"
        f"说明文件：{sidecar}\n"
        f"把这两个文件拷回本机，然后：\n"
        f"  python scripts/build_poetry_index.py --vectors {Path(args.out).name} "
        f"--chunks {Path(chunks).name}\n"
        f"耗时 {time.time() - started:.0f} 秒",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
