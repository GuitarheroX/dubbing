"""现代诗语料（``sheepzh/poetry``）的解析与切块。

规则（见 ``docs/rag-workflow.md`` 第 7 节）：

- 一个**诗节**（stanza，正文里被空行分开的一段）= 一个 chunk，**不设重叠**；
- 每个 chunk 的 token 数上限 256，用建库模型的 tokenizer 数；
- 超长的诗节先按**行**贪心打包，单行仍超长时按标点切，标点切完还超长才逐字硬切；
- ``title`` 等进 metadata，向量只放诗节正文；
- 行内换行保留——现代诗的断行本身就是句式。

本模块只用标准库，方便单测与离线统计，不依赖 chromadb / transformers。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "MIN_BODY_CHARS",
    "Poem",
    "Chunk",
    "TokenCounter",
    "corpus_root",
    "iter_poem_paths",
    "parse_poem",
    "poem_fingerprint",
    "iter_chunks",
    "count_tokens_by_chars",
]

#: 一个 chunk 的 token 上限；bge 系的输入上限是 512，留一倍余量。
DEFAULT_MAX_TOKENS = 256
#: 正文短于这个字数就不当诗收（残留标记、残句等）。
MIN_BODY_CHARS = 10

TokenCounter = Callable[[str], int]

_STANZA_SEP = re.compile(r"\n\s*\n")
#: 在标点之后切开，标点留在前一片里。
_FRAG_SEP = re.compile(r"(?<=[，。！？；：、,.!?;:])")


@dataclass(frozen=True)
class Poem:
    """一首诗：metadata + 逐节原文。"""

    poem_id: str
    poet: str
    title: str
    date: str
    stanzas: tuple[str, ...]

    @property
    def body(self) -> str:
        return "\n\n".join(self.stanzas)


@dataclass(frozen=True)
class Chunk:
    """一个待入库的片段：正好一个诗节，除非它太长被拆成了几份。"""

    chunk_id: str
    poem_id: str
    poet: str
    title: str
    date: str
    stanza_idx: int  # 1 起
    part_idx: int  # 1 起
    n_parts: int
    text: str
    chars: int
    tokens: int

    @property
    def continued(self) -> bool:
        """是否为同一诗节被拆开后的后续片段。"""
        return self.part_idx > 1

    def metadata(self) -> dict[str, str | int | bool]:
        """Chroma 的 metadata 只接受标量。"""
        return {
            "chunk_id": self.chunk_id,
            "poem_id": self.poem_id,
            "poet": self.poet,
            "title": self.title,
            "date": self.date,
            "stanza_idx": self.stanza_idx,
            "part_idx": self.part_idx,
            "n_parts": self.n_parts,
            "chars": self.chars,
            "tokens": self.tokens,
            "continued": self.continued,
        }

    def to_json(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "poem_id": self.poem_id,
            "poet": self.poet,
            "title": self.title,
            "date": self.date,
            "stanza_idx": self.stanza_idx,
            "part_idx": self.part_idx,
            "n_parts": self.n_parts,
            "text": self.text,
            "chars": self.chars,
            "tokens": self.tokens,
        }


# --------------------------------------------------------------------------
# 语料定位与解析
# --------------------------------------------------------------------------


def corpus_root(path: Path | str) -> Path:
    """把「仓库根」「仓库根/data」或「诗人目录的父目录」统一成后者。

    ``sheepzh/poetry`` 的目录形状是 ``<root>/data/<诗人名>_<拼音>/<诗标题>.pt``。
    """
    p = Path(path)
    nested = p / "data"
    if nested.is_dir() and any(nested.glob("*/*.pt")):
        return nested
    return p


def iter_poem_paths(root: Path | str) -> Iterator[Path]:
    """按诗人目录顺序产出 ``.pt`` 文件路径。"""
    base = corpus_root(root)
    for poet_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for path in sorted(poet_dir.glob("*.pt")):
            yield path


def _normalize(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\u3000", " ")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def _split_header(text: str) -> tuple[str, str, str]:
    """切出 ``title:`` / ``date:`` 头，返回 ``(title, date, body)``。

    语料格式是一段可选的 header、一个空行、然后是正文。第一行不是 ``title:`` / ``date:``
    时按"没有 header"处理，整份文本都是正文。
    """
    lines = text.split("\n")
    title = date = ""
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            break
        if line.startswith("title:"):
            title = line[len("title:") :].strip()
        elif line.startswith("date:"):
            date = line[len("date:") :].strip()
        else:
            break
        i += 1
    return title, date, "\n".join(lines[i:]).strip()


def parse_poem(path: Path | str) -> Poem | None:
    """解析一个 ``.pt`` 文件；太短、空正文、读不出内容时返回 ``None``。"""
    path = Path(path)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    title, date, body = _split_header(_normalize(raw))
    stanzas = tuple(s.strip() for s in _STANZA_SEP.split(body) if s.strip())
    if not stanzas:
        return None
    if sum(1 for ch in body if not ch.isspace()) < MIN_BODY_CHARS:
        return None
    poet_dir = path.parent.name
    poet = poet_dir.rsplit("_", 1)[0] if "_" in poet_dir else poet_dir
    return Poem(
        poem_id=f"{poet_dir}/{path.stem}",
        poet=poet,
        title=title or path.stem,
        date=date,
        stanzas=stanzas,
    )


def poem_fingerprint(poem: Poem) -> str:
    """正文指纹，用来砍跨诗人的完全重复。"""
    body = re.sub(r"\s+", "", poem.body)
    return hashlib.sha1(body.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# 切块
# --------------------------------------------------------------------------


def count_tokens_by_chars(text: str) -> int:
    """``transformers`` 不可用时的保守估算：非空白字符数。

    中文 BERT 词表里一个字通常就是一个 token（部分双字词会合并），所以"字符数"
    是该模型 token 数的**上界**，用它判 256 上限不会放超长片段进去。
    """
    return sum(1 for ch in text if not ch.isspace())


def _count_chars(text: str) -> int:
    return sum(1 for ch in text if not ch.isspace())


def _pack(units: list[str], count_tokens: TokenCounter, max_tokens: int, joiner: str) -> list[str]:
    """贪心打包已经"单独放得下"的单位。"""
    out: list[str] = []
    buf: list[str] = []
    for unit in units:
        if buf and count_tokens(joiner.join([*buf, unit])) > max_tokens:
            out.append(joiner.join(buf))
            buf = [unit]
        else:
            buf.append(unit)
    if buf:
        out.append(joiner.join(buf))
    return out


def _hard_split(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """逐字硬切，保证每片都不超上限（只用于病态长行）。"""
    out: list[str] = []
    buf = ""
    for ch in text:
        if buf and count_tokens(buf + ch) > max_tokens:
            out.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _split_long_line(line: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """把一条超长行按标点切开再打包；拼接回去与原文完全一致。"""
    pieces: list[str] = []
    for frag in (f for f in _FRAG_SEP.split(line) if f):
        if count_tokens(frag) <= max_tokens:
            pieces.append(frag)
        else:
            pieces.extend(_hard_split(frag, count_tokens, max_tokens))
    return _pack(pieces, count_tokens, max_tokens, "")


def _split_stanza(text: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            parts.extend(_pack(pending, count_tokens, max_tokens, "\n"))
            pending.clear()

    for line in text.split("\n"):
        if count_tokens(line) <= max_tokens:
            pending.append(line)
            continue
        flush()
        parts.extend(_split_long_line(line, count_tokens, max_tokens))
    flush()
    return parts or [text]


def _chunk_id(poem: Poem, stanza_idx: int, part_idx: int, n_parts: int) -> str:
    base = f"{poem.poem_id}#s{stanza_idx}"
    return base if n_parts == 1 else f"{base}p{part_idx}"


def iter_chunks(
    poem: Poem,
    *,
    count_tokens: TokenCounter = count_tokens_by_chars,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Iterator[Chunk]:
    """把一个诗节切成一个 chunk；超长的才拆，拆出来的块共享同一份 metadata。"""
    for stanza_idx, stanza in enumerate(poem.stanzas, 1):
        parts = _split_stanza(stanza, count_tokens, max_tokens)
        n_parts = len(parts)
        for part_idx, text in enumerate(parts, 1):
            yield Chunk(
                chunk_id=_chunk_id(poem, stanza_idx, part_idx, n_parts),
                poem_id=poem.poem_id,
                poet=poem.poet,
                title=poem.title,
                date=poem.date,
                stanza_idx=stanza_idx,
                part_idx=part_idx,
                n_parts=n_parts,
                text=text,
                chars=_count_chars(text),
                tokens=count_tokens(text),
            )
