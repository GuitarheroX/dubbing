#!/usr/bin/env python3
"""建本地现代诗索引：语料 → 诗节切块 → 向量 → 本地 Chroma。

**这条链路分两台机器**，本机不需要 GPU，也不需要起任何嵌入服务：

1. 【本机】递归读 ``sheepzh/poetry`` 的 ``data/<诗人>_<拼音>/<诗标题>.pt``，按**诗节**切块
   （一个诗节一个 chunk、不重叠、超长才拆、上限 ``--max-tokens``），写 ``chunks.jsonl``；
2. 【本机 → 服务器】把仓库和 ``chunks.jsonl`` 拷上去；
3. 【服务器】``scripts/embed_poetry.py`` 用 GPU 跑到 ``vectors.npy`` + ``vectors.json``
   （只依赖根目录 ``requirements.txt``）；
4. 【服务器 → 本机】把这两个文件拷回来；
5. 【本机】``--vectors vectors.npy`` 把它们灌进本地 Chroma，并写 ``manifest.json``。

目录组织（``--out`` 默认 ``data/poetry``）：

    data/poetry/
      raw/              语料 clone（gitignore）
      chunks.jsonl      ① 切块中间产物 —— 要上传的文件
      vectors.npy       ③ 服务器产出的向量 —— 要下载的文件
      vectors.json      ③ 配对说明（模型、维度、条数、chunks 的 sha1）
      chroma/           ⑤ 本地向量库
      manifest.json     ⑤ 索引血缘（唯一建议入库的一个）

``vectors.json`` 里记着 chunks 文件的 sha1，灌库时会核对，防止"换了切块参数但向量还是旧的"。

如果本机能直接连到 GPU 服务器，也可以走另一种模式：不给 ``--vectors``，让脚本边切边调
OpenAI 兼容的嵌入端点（TEI / vLLM）。两种模式产出的 Chroma 索引没有区别。

用法：

    # ① 先看能不能切、切出多少块（不联网、不建库）
    python scripts/build_poetry_index.py --poetry-root data/poetry/raw --dry-run

    # ③④ 服务器上跑向量（详见 scripts/embed_poetry.py 头部注释与 requirements.txt）

    # ⑤ 灌库 + 抽查 5 条，确认服务器与本机算出来的向量一致
    python scripts/build_poetry_index.py --vectors data/poetry/vectors.npy --verify 5 --reset

    # 备选：直接连嵌入服务（有 GPU 服务且网络可达时）
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
    l2_normalize,
)
from dubbing.rag.retriever import DEFAULT_COLLECTION  # noqa: E402
from dubbing.rag.store import ChromaStore, StoreError, write_manifest  # noqa: E402
from dubbing.rag.vectors import (  # noqa: E402
    VectorFileError,
    count_lines,
    file_sha1,
    load_vectors,
)

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


class _Progress:
    """每 PROGRESS_EVERY 条报一次进度。"""

    def __init__(self, total: int, started: float, quiet: bool) -> None:
        self.total = total
        self.started = started
        self.quiet = quiet
        self.next_at = PROGRESS_EVERY

    def tick(self, done: int) -> None:
        if self.quiet or done < self.next_at:
            return
        rate = done / max(time.time() - self.started, 1e-6)
        print(f"  已入库 {done}/{self.total}（{rate:.0f} 块/秒）", file=sys.stderr, flush=True)
        self.next_at += PROGRESS_EVERY


def _ingest_online(
    store: ChromaStore,
    chunks_path: Path,
    embedding,
    *,
    batch_size: int,
    total: int,
    started: float,
    quiet: bool,
) -> tuple[int, int | None]:
    """向量现算：边读 chunks.jsonl 边调嵌入服务，然后 upsert。"""
    written = 0
    dimension: int | None = None
    buffer: list[Chunk] = []
    progress = _Progress(total, started, quiet)

    def flush(items: list[Chunk]) -> int:
        nonlocal dimension
        texts = [c.text for c in items]
        vectors = embedding.embed(texts)
        if len(vectors) != len(items):
            raise EmbeddingError(f"嵌入服务返回 {len(vectors)} 条，预期 {len(items)} 条")
        if dimension is None and vectors:
            dimension = len(vectors[0])
        return store.upsert(
            ids=[c.chunk_id for c in items],
            documents=texts,
            metadatas=[c.metadata() for c in items],
            embeddings=vectors,
        )

    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            buffer.append(Chunk(**json.loads(line)))
            if len(buffer) < batch_size:
                continue
            written += flush(buffer)
            buffer.clear()
            progress.tick(written)
    if buffer:
        written += flush(buffer)
        progress.tick(written)
    return written, dimension


def _ingest_offline(
    store: ChromaStore,
    chunks_path: Path,
    vectors,
    *,
    batch_size: int,
    normalize: bool,
    total: int,
    started: float,
    quiet: bool,
) -> tuple[int, int | None]:
    """向量已算好（服务器产出）：只读文件 + upsert，不碰任何嵌入服务。

    ``vectors`` 用 ``mmap_mode="r"`` 打开，所以本机内存占用与向量文件大小无关。
    """
    written = 0
    progress = _Progress(total, started, quiet)
    with chunks_path.open("r", encoding="utf-8") as fh:
        rows: list[Chunk] = []
        for line in fh:
            rows.append(Chunk(**json.loads(line)))
            if len(rows) < batch_size:
                continue
            start = written
            written += _upsert_slice(
                store, rows, vectors, start=start, normalize=normalize
            )
            rows.clear()
            progress.tick(written)
        if rows:
            written += _upsert_slice(store, rows, vectors, start=written, normalize=normalize)
            progress.tick(written)
    return written, int(vectors.shape[1]) if written else None


def _upsert_slice(
    store: ChromaStore,
    rows: list[Chunk],
    vectors,
    *,
    start: int,
    normalize: bool,
) -> int:
    block = vectors[start : start + len(rows)]
    batch = [row.astype("float32").tolist() for row in block]
    if normalize:
        batch = [l2_normalize(v) for v in batch]
    return store.upsert(
        ids=[c.chunk_id for c in rows],
        documents=[c.text for c in rows],
        metadatas=[c.metadata() for c in rows],
        embeddings=batch,
    )


def _verify_vectors(
    chunks_path: Path,
    vectors,
    *,
    model: str,
    samples: int,
    device: str | None,
) -> int:
    """抽查：本机现算的向量与服务器给的向量对不对得上。

    这是"服务器出向量、本机查询"这条分工唯一真正的风险点——两边模型/池化不一致时，
    检索会静默变差。余弦相似度应当接近 1。
    """
    import random

    from dubbing.rag.embedding import LocalEmbeddingClient

    total = int(vectors.shape[0])
    count = min(samples, total)
    indices = sorted(random.Random(0).sample(range(total), count))
    wanted = set(indices)
    texts: list[str] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i in wanted:
                texts.append(json.loads(line)["text"])
            if len(texts) == count:
                break

    encoder = LocalEmbeddingClient(model, device=device, batch_size=count)
    local = encoder.embed(texts)

    print(f"\n抽查 {count} 条：本机现算 vs 服务器给的向量（模型 {model}）", file=sys.stderr)
    worst = 1.0
    for index, text, vector in zip(indices, texts, local):
        remote = vectors[index].astype("float32")
        dot = sum(a * b for a, b in zip(vector, remote.tolist()))
        norm = (sum(x * x for x in remote.tolist()) ** 0.5) or 1.0
        similarity = dot / norm  # 本机向量已归一化
        worst = min(worst, similarity)
        head = text.replace("\n", " ")[:24]
        print(f"  [{similarity:.4f}] {head}…", file=sys.stderr)
    if worst < 0.99:
        print(
            f"\n警告：最低相似度只有 {worst:.4f}。两边很可能不是同一个模型或同一种池化，"
            "检索质量会变差。请核对 --model 与服务器上用的模型。",
            file=sys.stderr,
        )
        return 1
    print(f"一致（最低 {worst:.4f}）", file=sys.stderr)
    return 0


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
    parser.add_argument(
        "--chunks",
        default=None,
        help="切块文件，默认 <out>/chunks.jsonl；用 --vectors 时必须与向量同源",
    )
    parser.add_argument(
        "--vectors",
        default=None,
        help="离线向量文件（服务器上 embed_poetry.py 产出的 .npy）。给了它就不调嵌入服务，也不需要 GPU",
    )
    parser.add_argument(
        "--verify",
        type=int,
        default=0,
        metavar="N",
        help="离线模式下抽查 N 条：本机现算向量与文件里的比对（需要本机有 sentence-transformers）",
    )
    parser.add_argument("--reset", action="store_true", help="先清空 collection 再写")
    parser.add_argument("--dry-run", action="store_true", help="只解析 + 切块 + 统计，不嵌入、不建库")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_tokens < 1 or args.batch < 1:
        print("--max-tokens 与 --batch 必须是正整数", file=sys.stderr)
        return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = Path(args.chunks) if args.chunks else out_dir / "chunks.jsonl"
    started = time.time()
    tokenizer_name = ""
    stats: dict[str, int] = {}
    total_chunks = 0
    offline = bool(args.vectors)

    if offline:
        if args.dry_run:
            print(
                "--dry-run 是给切块用的；已经有向量文件时请直接灌库（去掉 --dry-run），"
                "想核对向量就用 --verify N。",
                file=sys.stderr,
            )
            return 2
        if not chunks_path.is_file():
            print(
                f"找不到切块文件 {chunks_path}。离线模式需要与向量同源的 chunks.jsonl，"
                "用 --chunks 指过去。",
                file=sys.stderr,
            )
            return 1
        total_chunks = count_lines(chunks_path)
        print(
            f"离线模式：{chunks_path}（{total_chunks} 条）+ {args.vectors}\n"
            "不解析语料、不调嵌入服务。",
            file=sys.stderr,
        )
    else:
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

        count_tokens, tokenizer_name = load_token_counter(args.tokenizer or args.embed_model)
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
        total_chunks = stats["chunks"]
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

    embedding = None
    vectors = None
    vector_meta = None
    dimension: int | None = None

    if offline:
        try:
            vectors, vector_meta = load_vectors(
                args.vectors,
                expected_count=total_chunks,
                chunks_file=chunks_path,
                chunks_sha1=file_sha1(chunks_path),
                mmap=True,
            )
        except (VectorFileError, ValueError) as exc:
            print(f"向量文件不可用：{exc}", file=sys.stderr)
            return 1
        model_name = vector_meta.model or args.embed_model
        dimension = vector_meta.dim or int(vectors.shape[1])
        source = "vectors"
        print(
            f"向量：{vectors.shape[0]} 条 × {vectors.shape[1]} 维"
            f"（{vector_meta.dtype}，模型 {model_name}）\n"
            f"开始灌库（不调嵌入服务）：每批 {args.batch}",
            file=sys.stderr,
        )
    else:
        try:
            embedding = build_embedding_client(
                url=args.embed_url,
                model=args.embed_model,
                api_key=args.api_key,
                local=args.embed_local,
            )
        except EmbeddingError as exc:
            print(f"造不出嵌入客户端：{exc}", file=sys.stderr)
            return 1
        # manifest 与 index_id 记的是**真正用来嵌入的模型**（客户端可被显式指定覆盖）
        model_name = getattr(embedding, "model", None) or args.embed_model
        source = getattr(embedding, "name", "embedding")
        print(
            f"开始嵌入：{total_chunks} 块 → {embedding.name}({embedding.model})"
            f"，每批 {args.batch}",
            file=sys.stderr,
        )

    index_id = _index_id(args.collection, model_name, args.max_tokens, args.index_id)

    written = 0
    try:
        if offline:
            written, dimension = _ingest_offline(
                store,
                chunks_path,
                vectors,
                batch_size=args.batch,
                normalize=not vector_meta.normalize,
                total=total_chunks,
                started=started,
                quiet=args.quiet,
            )
        else:
            written, dimension = _ingest_online(
                store,
                chunks_path,
                embedding,
                batch_size=args.batch,
                total=total_chunks,
                started=started,
                quiet=args.quiet,
            )
    except (EmbeddingError, StoreError, VectorFileError) as exc:
        print(f"\n建库中断：{exc}\n已写入 {written} 条，修好后重跑即可（upsert 覆盖）。", file=sys.stderr)
        return 1

    corpus = (
        {
            "chunks_file": str(chunks_path),
            "chunks": total_chunks,
            "sha1": vector_meta.chunks_sha1 if vector_meta else "",
        }
        if offline
        else {
            "path": str(root),
            "poems": stats["poems"],
            "skipped": stats["skipped"],
            "duplicates": stats["duplicates"],
            "chunks": stats["chunks"],
        }
    )
    manifest = {
        "index_id": index_id,
        "collection": args.collection,
        "db": str(db_dir),
        "source": source,
        "corpus": corpus,
        "model": {
            "name": model_name,
            "dim": dimension,
            "pooling": vector_meta.pooling if offline and vector_meta else "cls",
            "normalize": True,
            "instruction": None,
        },
        "chunking": {
            "unit": "stanza",
            "max_tokens": args.max_tokens,
            "overlap": 0,
            "tokenizer": tokenizer_name or ("unknown(offline)" if offline else ""),
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

    if offline and args.verify > 0:
        try:
            return _verify_vectors(
                chunks_path, vectors, model=model_name, samples=args.verify, device=None
            )
        except EmbeddingError as exc:
            print(
                f"\n抽查需要本机也能加载同一个模型：{exc}\n"
                "（索引已经建好了，这条只是校验；本机装了 sentence-transformers 就能跑）",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
