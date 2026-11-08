#!/usr/bin/env python3
"""建本地现代诗索引：语料 → 诗节切块 → GPU 文本嵌入 → 本地 Chroma。

全程只用"本机磁盘 + 一个嵌入端点"：

1. 递归读 ``sheepzh/poetry`` 的 ``data/<诗人>_<拼音>/<诗标题>.pt``；
2. 按**诗节**切块（一个诗节一个 chunk，不重叠，超长才拆，上限 ``--max-tokens``）；
3. 切块结果写 ``<out>/chunks.jsonl`` —— 中间产物，之后换嵌入模型或重建索引都不必重解析语料；
4. 分批把正文发到嵌入服务（GPU 服务器上的 TEI / vLLM，OpenAI 兼容），向量写本地 Chroma；
5. 写 ``<out>/manifest.json``（index_id、模型、切块参数、语料规模），供管线记录血缘。

用法：

    # 先看能不能切、切出多少块（不联网、不建库）
    python scripts/build_poetry_index.py --poetry-root data/poetry/raw --dry-run

    # 用子集跑通
    python scripts/build_poetry_index.py --poetry-root data/poetry/raw --limit 20000

    # 全量 + 重建
    DUBBING_EMBED_URL=http://gpu:8080/v1/embeddings \\
      python scripts/build_poetry_index.py --poetry-root data/poetry/raw --reset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.rag.chunking import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    Chunk,
    count_tokens_by_chars,
    corpus_root,
    iter_chunks,
    iter_poem_paths,
    parse_poem,
    poem_fingerprint,
)
from dubbing.rag.embedding import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    EmbeddingError,
    build_embedding_client,
)
from dubbing.rag.retriever import DEFAULT_COLLECTION  # noqa: E402
from dubbing.rag.store import ChromaStore, StoreError, write_manifest  # noqa: E402

__all__ = ["main", "load_token_counter", "collect_chunks"]

PROGRESS_EVERY = 20000


def load_token_counter(model: str) -> tuple[Callable[[str], int], str]:
    """优先用建库模型的 tokenizer 数 token；拿不到就退回字符数上界。

    拿不到的情形有两种：没装 ``transformers``，或者模型名在 HuggingFace 上取不到
    （自建模型名、离线机器）。两种都不该让建库直接崩——字符数在中文里是 token 数的上界，
    用它判 256 上限是安全的（只会切得略碎，不会放超长片段进索引）。
    """
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        print(
            "警告：没装 transformers，token 数改用「非空白字符数」保守估算"
            "（中文里它 ≥ 真实 token 数，所以不会放超长片段进索引）。",
            file=sys.stderr,
        )
        return count_tokens_by_chars, "chars(fallback)"

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as exc:  # 模型名不在 HF、离线、没有缓存……
        print(
            f"警告：取不到 {model} 的 tokenizer（{type(exc).__name__}: {exc}）；"
            "token 数改用「非空白字符数」保守估算。想用真 tokenizer 就加 --tokenizer <HF 模型名>。",
            file=sys.stderr,
        )
        return count_tokens_by_chars, "chars(fallback)"

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count, f"transformers:{model}"


def collect_chunks(
    root: Path,
    *,
    count_tokens: Callable[[str], int],
    max_tokens: int,
    limit: int | None,
    quiet: bool = False,
) -> tuple[list[Chunk], dict[str, int]]:
    """解析 + 切块 + 去重。返回 ``(chunks, stats)``。

    语料量级（80K 首、15M 字）在这个函数里被一次性读进内存是可行的：
    切完约 30–40 万块，每块几十字节到几百字节。
    """
    stats = {"poems": 0, "skipped": 0, "duplicates": 0, "chunks": 0, "oversize_parts": 0}
    chunks: list[Chunk] = []
    seen: set[str] = set()

    for path in iter_poem_paths(root):
        if limit is not None and stats["poems"] >= limit:
            break
        poem = parse_poem(path)
        if poem is None:
            stats["skipped"] += 1
            continue
        fingerprint = poem_fingerprint(poem)
        if fingerprint in seen:
            stats["duplicates"] += 1
            continue
        seen.add(fingerprint)
        stats["poems"] += 1
        for chunk in iter_chunks(poem, count_tokens=count_tokens, max_tokens=max_tokens):
            chunks.append(chunk)
            if chunk.n_parts > 1:
                stats["oversize_parts"] += 1
        if not quiet and stats["poems"] % PROGRESS_EVERY == 0:
            print(
                f"  已解析 {stats['poems']} 首 / {len(chunks)} 块",
                file=sys.stderr,
                flush=True,
            )

    stats["chunks"] = len(chunks)
    return chunks, stats


def _index_id(collection: str, model: str, max_tokens: int, override: str | None) -> str:
    if override:
        return override
    slug = model.rstrip("/").split("/")[-1]
    return f"poetry_stanza{max_tokens}_{slug}_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="建 workflow-RAG 用的本地现代诗索引")
    parser.add_argument(
        "--poetry-root",
        default=str(REPO_ROOT / "data" / "poetry" / "raw"),
        help="sheepzh/poetry 的仓库根，或它里面的 data/ 目录",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "poetry"), help="索引与中间产物的输出目录")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--index-id", default=None, help="默认按 collection + 模型 + 上限自动生成")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 首诗（跑通或建子集索引用）")
    parser.add_argument("--batch", type=int, default=128, help="一次嵌入请求的条数")
    parser.add_argument("--tokenizer", default=None, help="数 token 用的模型，默认与嵌入模型相同")
    parser.add_argument("--embed-url", default=None)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--embed-local", action="store_true", help="本机推理（需要 sentence-transformers）")
    parser.add_argument("--reset", action="store_true", help="先清空 collection 再写")
    parser.add_argument("--dry-run", action="store_true", help="只解析 + 切块 + 统计，不嵌入、不建库")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens < 1 or args.batch < 1:
        print("--max-tokens 与 --batch 必须是正整数", file=sys.stderr)
        return 2

    root = Path(args.poetry_root)
    if not root.is_dir():
        print(
            f"找不到语料目录 {root}。先克隆语料：\n"
            f"  git clone --depth 1 https://github.com/sheepzh/poetry {root}",
            file=sys.stderr,
        )
        return 1
    root = corpus_root(root)
    if not any(root.glob("*/*.pt")):
        print(f"{root} 下没有 <诗人>/<诗>.pt，检查 --poetry-root 指向的是不是仓库的 data/ 目录", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "chunks.jsonl"

    count_tokens, tokenizer_name = load_token_counter(args.tokenizer or args.embed_model)
    started = time.time()
    print(f"解析语料：{root}", file=sys.stderr)
    chunks, stats = collect_chunks(
        root,
        count_tokens=count_tokens,
        max_tokens=args.max_tokens,
        limit=args.limit,
        quiet=args.quiet,
    )

    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_json(), ensure_ascii=False) + "\n")

    longest = max((c.tokens for c in chunks), default=0)
    print(
        f"切块完成：{stats['poems']} 首 → {stats['chunks']} 块"
        f"（跳过 {stats['skipped']} 首，去重 {stats['duplicates']} 首，"
        f"被拆开的诗节 {stats['oversize_parts']} 个，最长 {longest} token）\n"
        f"中间产物：{chunks_path}",
        file=sys.stderr,
    )

    if args.dry_run:
        index_id = _index_id(args.collection, args.embed_model, args.max_tokens, args.index_id)
        print(f"（--dry-run：没有嵌入、没有建库）index_id 会是 {index_id}", file=sys.stderr)
        return 0

    db_dir = out_dir / "chroma"
    try:
        store = ChromaStore(db_dir, args.collection, create=True)
    except StoreError as exc:
        print(f"打不开本地索引：{exc}", file=sys.stderr)
        return 1

    existing = store.count()
    if args.reset:
        store.reset()
        existing = 0
    elif existing:
        print(
            f"注意：collection 里已有 {existing} 条。同一 chunk_id 会被覆盖，"
            "但语料或切块参数变过就会留下旧块——建议加 --reset。",
            file=sys.stderr,
        )

    try:
        embedding = build_embedding_client(
            url=args.embed_url, model=args.embed_model, api_key=args.api_key, local=args.embed_local
        )
    except EmbeddingError as exc:
        print(f"造不出嵌入客户端：{exc}", file=sys.stderr)
        return 1

    # manifest 与 index_id 记的是**真正用来嵌入的模型**（客户端可被显式指定覆盖）
    model_name = getattr(embedding, "model", None) or args.embed_model
    index_id = _index_id(args.collection, model_name, args.max_tokens, args.index_id)

    print(
        f"开始嵌入：{len(chunks)} 块 → {embedding.name}({embedding.model})"
        f"，每批 {args.batch}",
        file=sys.stderr,
    )

    written = 0
    collected = 0
    dimension: int | None = None
    buffer: list[Chunk] = []
    next_report = PROGRESS_EVERY

    def flush_buffer(batch: list[Chunk]) -> int:
        nonlocal dimension
        texts = [c.text for c in batch]
        vectors = embedding.embed(texts)
        if len(vectors) != len(batch):
            raise EmbeddingError(f"嵌入服务返回 {len(vectors)} 条，预期 {len(batch)} 条")
        if dimension is None and vectors:
            dimension = len(vectors[0])
        return store.upsert(
            ids=[c.chunk_id for c in batch],
            documents=texts,
            metadatas=[c.metadata() for c in batch],
            embeddings=vectors,
        )

    try:
        with chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                buffer.append(Chunk(**json.loads(line)))
                if len(buffer) < args.batch:
                    continue
                written += flush_buffer(buffer)
                buffer.clear()
                collected = written
                if collected >= next_report:
                    rate = collected / max(time.time() - started, 1e-6)
                    print(
                        f"  已入库 {collected}/{len(chunks)}（{rate:.0f} 块/秒）",
                        file=sys.stderr,
                        flush=True,
                    )
                    next_report += PROGRESS_EVERY
        if buffer:
            written += flush_buffer(buffer)
    except (EmbeddingError, StoreError) as exc:
        print(f"\n建库中断：{exc}\n已写入 {written} 条，修好后重跑即可（upsert 覆盖）。", file=sys.stderr)
        return 1

    manifest = {
        "index_id": index_id,
        "collection": args.collection,
        "db": str(db_dir),
        "corpus": {
            "path": str(root),
            "poems": stats["poems"],
            "skipped": stats["skipped"],
            "duplicates": stats["duplicates"],
            "chunks": stats["chunks"],
        },
        "model": {
            "name": model_name,
            "dim": dimension,
            "pooling": "cls",
            "normalize": True,
            "instruction": None,
        },
        "chunking": {
            "unit": "stanza",
            "max_tokens": args.max_tokens,
            "overlap": 0,
            "tokenizer": tokenizer_name,
        },
        "stored": store.count(),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_manifest(db_dir, manifest)

    print(
        f"\n完成：{written} 块写入 {db_dir}（collection {args.collection}，共 {store.count()} 条）\n"
        f"index_id：{index_id}\nmanifest：{db_dir.parent / 'manifest.json'}\n"
        f"耗时 {time.time() - started:.0f} 秒",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
