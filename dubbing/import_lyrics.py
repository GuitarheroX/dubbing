"""把人工标注的歌词文本转换成 ``song.yaml``。

输入格式（空行分段，段内每行一句）::

    ## 歌名
    歌词 总字数 [分句...]
    ...
    歌词 *这句不用翻译*

- 每行第一个数字是这句要唱的字数（MVP 里 ``min == max``）；
- 后面的数字是 ``pattern``，即这一句在旋律上被切成几段、每段几个音；
- ``*...*`` 是给人看的批注；含「不用翻译」时该句标记为 ``keep_source``；
- 标题后可以写 ``[song-id]`` 显式指定 id，否则按内置表/歌名推导。

空行只决定 section 的边界，不改变乐句顺序。section 的 role 由本脚本按
位置与重复关系做一个粗略推测，导入后仍建议人工复核。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .models import SchemaError, Song
from .song import song_path

__all__ = [
    "LyricsImportError",
    "PhraseSpec",
    "Block",
    "ParsedSong",
    "parse_lyrics",
    "infer_roles",
    "build_song",
    "dump_song",
    "main",
]


class LyricsImportError(ValueError):
    """输入文本不符合约定格式，或数字之间自相矛盾。"""


#: 这些歌名对应的 song id。找不到时可以用标题后的 ``[id]`` 显式指定。
TITLE_IDS: dict[str, str] = {
    "ソナーレ": "sonare",
    "炉心融解": "roshin-yukai",
    "忘れてください": "wasurete-kudasai",
    "栞": "shiori",
    "LOVE 2000": "love-2000",
    "さよならペンギン": "sayonara-penguin",
    "ふうせん": "fuusen",
    "向日葵": "himawari",
    "夜明けと蛍": "yoake-to-hotaru",
}

_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
_TRAILING_COMMENT = re.compile(r"\s*\*(?P<comment>[^*]+)\*\s*$")
_LINE = re.compile(r"^(?P<ja>.+?)\s+(?P<nums>\d+(?:\s+\d+)*)$")
_TITLE_ID = re.compile(r"^(?P<title>.+?)\s*\[(?P<id>[a-z0-9][a-z0-9-]*)\]\s*$")

_YAML_WORDS = {"null", "true", "false", "yes", "no", "on", "off", "~"}


@dataclass(frozen=True)
class PhraseSpec:
    ja: str
    #: 这一句要唱几个字；``keep_source`` 时为 ``None``。
    budget: int | None = None
    pattern: tuple[int, ...] | None = None
    keep_source: bool = False
    notes: str = ""


@dataclass
class Block:
    """输入里由空行隔开的一组乐句。"""

    phrases: list[PhraseSpec] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.phrases)


@dataclass
class ParsedSong:
    title: str
    song_id: str
    blocks: list[Block]
    warnings: list[str] = field(default_factory=list)

    def role_of_blocks(self) -> list[str]:
        return infer_roles(self.blocks)


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------


def _note(*parts: str) -> str:
    return "；".join(p for p in parts if p)


def _parse_phrase(line: str, lineno: int, *, strict: bool) -> tuple[PhraseSpec, list[str]]:
    warnings: list[str] = []
    raw = line.strip()
    note = ""
    m = _TRAILING_COMMENT.search(raw)
    if m:
        note = m.group("comment").strip()
        raw = raw[: m.start()].strip()

    if not raw:
        raise LyricsImportError(f"第 {lineno} 行：乐句是空的")

    keep_source = "不用翻译" in note or "保留原文" in note
    m = _LINE.match(raw)
    if m is None:
        if keep_source:
            return PhraseSpec(ja=raw, keep_source=True, notes=note), warnings
        raise LyricsImportError(
            f"第 {lineno} 行：找不到行尾的总字数（格式应为「歌词 总字数 [分句...]」）：{line!r}"
        )

    ja = m.group("ja").strip()
    if not ja:
        raise LyricsImportError(f"第 {lineno} 行：乐句是空的")

    nums = [int(x) for x in m.group("nums").split()]
    total = nums[0]
    parts = nums[1:]
    if total < 1:
        raise LyricsImportError(f"第 {lineno} 行：总字数必须为正整数，得到 {total}")

    if keep_source:
        warnings.append(
            f"第 {lineno} 行：{ja!r} 保留原文，已忽略其字数 {total} 与分句 {parts}"
        )
        return PhraseSpec(ja=ja, keep_source=True, notes=note), warnings

    pattern: tuple[int, ...] | None = None
    extra_note = ""
    if parts:
        bad = [p for p in parts if p < 1]
        if bad:
            raise LyricsImportError(f"第 {lineno} 行：分句必须是正整数，得到 {parts}")
        actual = sum(parts)
        if actual == total:
            pattern = tuple(parts)
        elif actual < total:
            missing = total - actual
            pattern = tuple(parts + [missing])
            extra_note = f"分句 {parts} 合计 {actual}，已补成 {list(pattern)}"
            warnings.append(f"第 {lineno} 行：{extra_note}")
            if strict:
                raise LyricsImportError(f"第 {lineno} 行：{extra_note}")
        else:
            extra_note = f"分句 {parts} 合计 {actual} 超过总字数 {total}，已忽略分句"
            warnings.append(f"第 {lineno} 行：{extra_note}")
            if strict:
                raise LyricsImportError(f"第 {lineno} 行：{extra_note}")
            pattern = None

    # [N] 与“不写 pattern”等价，省掉这个冗余字段。
    if pattern is not None and len(pattern) == 1 and pattern[0] == total:
        pattern = None

    return (
        PhraseSpec(
            ja=ja,
            budget=total,
            pattern=pattern,
            notes=_note(note, extra_note),
        ),
        warnings,
    )


def _split_title(title: str) -> tuple[str, str | None]:
    m = _TITLE_ID.match(title)
    if m:
        return m.group("title").strip(), m.group("id")
    return title, None


def _ascii_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug


def _song_id(title: str, explicit: str | None, index: int, warnings: list[str]) -> str:
    if explicit:
        return explicit
    if title in TITLE_IDS:
        return TITLE_IDS[title]
    slug = _ascii_slug(title)
    if slug:
        return slug
    fallback = f"song-{index:02d}"
    warnings.append(
        f"歌名 {title!r} 没有对应的 song id，暂用 {fallback}；"
        "可在标题后写 [song-id] 指定"
    )
    return fallback


def parse_lyrics(text: str, *, strict: bool = False) -> list[ParsedSong]:
    """把整段标注文本解析成若干首歌。"""

    songs: list[ParsedSong] = []
    used_ids: set[str] = set()

    title: str | None = None
    explicit_id: str | None = None
    blocks: list[Block] = []
    current: list[PhraseSpec] = []
    warnings: list[str] = []

    def flush_block() -> None:
        nonlocal current
        if current:
            blocks.append(Block(current))
            current = []

    def finish_song() -> None:
        nonlocal title, explicit_id, blocks, current, warnings
        if title is None:
            return
        flush_block()
        if not blocks:
            raise LyricsImportError(f"歌曲 {title!r} 没有任何乐句")
        sid = _song_id(title, explicit_id, len(songs) + 1, warnings)
        if sid in used_ids:
            base = sid
            k = 2
            while f"{base}-{k}" in used_ids:
                k += 1
            sid = f"{base}-{k}"
            warnings.append(f"song id {base!r} 重复，改用 {sid!r}")
        used_ids.add(sid)
        songs.append(ParsedSong(title=title, song_id=sid, blocks=blocks, warnings=warnings))
        title = None
        explicit_id = None
        blocks = []
        current = []
        warnings = []

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            flush_block()
            continue

        m = _HEADING.match(line)
        if m:
            finish_song()
            title, explicit_id = _split_title(m.group("title").strip())
            if not title:
                raise LyricsImportError(f"第 {lineno} 行：歌曲标题是空的")
            continue

        if title is None:
            warnings.append(f"第 {lineno} 行：第一个歌曲标题之前的内容已忽略：{line!r}")
            continue

        phrase, ws = _parse_phrase(line, lineno, strict=strict)
        current.append(phrase)
        warnings.extend(ws)

    finish_song()
    if not songs:
        raise LyricsImportError("没有找到任何以 # 或 ## 开头的歌曲标题")
    return songs


# --------------------------------------------------------------------------
# section 与 song.yaml
# --------------------------------------------------------------------------


def _similarity(a: Block, b: Block) -> float:
    la = {p.ja for p in a.phrases}
    lb = {p.ja for p in b.phrases}
    if not la or not lb:
        return 0.0
    return len(la & lb) / max(len(la), len(lb))


def infer_roles(blocks: list[Block]) -> list[str]:
    """按位置与重复关系粗略推测每段的 role。

    这不是音乐分析，只是给 A / B 一个不空的提示：
    最后一段通常是副歌；更长的收尾段的后一段、以及被明显重复的段，
    也按副歌处理；副歌前较短的段算 pre-chorus；最后 1–2 句的残段算 outro。
    """

    n = len(blocks)
    if n == 0:
        return []
    if n == 1:
        return ["chorus"]

    lengths = [len(b) for b in blocks]
    roles = [""] * n

    # 1) 最后一段：太短就视为 outro，副歌落在它前面最后一个“像样”的段。
    chorus_index: int | None
    if lengths[-1] <= 2:
        roles[-1] = "outro"
        chorus_index = None
        for i in range(n - 2, -1, -1):
            if lengths[i] > 2:
                chorus_index = i
                break
        if chorus_index is None:
            chorus_index = n - 2
        roles[chorus_index] = "chorus"
    else:
        roles[-1] = "chorus"
        chorus_index = n - 1

    # 2) 很长的倒数第二段，往往也是副歌（副歌连唱两段）。
    if n >= 3 and chorus_index == n - 1 and lengths[-2] >= 8:
        roles[-2] = "chorus"

    # 3) 两段高度重复时，通常都是副歌（主歌很少逐句重复）。
    #    这里必须两两比较，不能只和已经认定的副歌比：第一次出现的副歌
    #    在比较时还不知道自己也是副歌。
    for i in range(n):
        if roles[i] == "outro" or lengths[i] < 2:
            continue
        for j in range(i + 1, n):
            if roles[j] == "outro" or lengths[j] < 2:
                continue
            shorter = min(lengths[i], lengths[j])
            longer = max(lengths[i], lengths[j])
            if shorter / longer < 0.6:
                continue
            if _similarity(blocks[i], blocks[j]) >= 0.6:
                roles[i] = "chorus"
                roles[j] = "chorus"

    for i in range(n):
        if not roles[i]:
            roles[i] = "verse"

    # 4) 副歌前较短的段算 pre-chorus。第一段始终当 verse，否则一首只有
    #    两段的短歌会没有主歌。
    for i in range(1, n - 1):
        if roles[i] != "verse" or roles[i + 1] != "chorus":
            continue
        if lengths[i] > 4:
            continue
        # 2 句的引子后面接主歌，不把主歌误标成 pre-chorus。
        if i == 1 and lengths[0] <= 2:
            continue
        roles[i] = "pre-chorus"

    return roles


def build_song(parsed: ParsedSong) -> Song:
    roles = infer_roles(parsed.blocks)
    prefixes = {"verse": "V", "pre-chorus": "B", "chorus": "C", "outro": "O"}
    counts = {role: 0 for role in prefixes}

    sections: list[dict[str, str]] = []
    phrases: list[dict[str, object]] = []

    for block, role in zip(parsed.blocks, roles):
        if role not in prefixes:
            raise LyricsImportError(f"内部错误：未知 role {role!r}")
        counts[role] += 1
        sid = f"{prefixes[role]}{counts[role]}"
        sections.append({"id": sid, "role": role})

        width = max(2, len(str(len(block))))
        for i, p in enumerate(block.phrases, 1):
            pid = f"{sid}-{i:0{width}d}"
            row: dict[str, object] = {"id": pid, "section": sid, "ja": p.ja}
            if p.keep_source:
                row["keep_source"] = True
            else:
                if p.budget is None:
                    raise LyricsImportError(f"内部错误：{pid} 没有 budget")
                row["budget"] = {"min": p.budget, "max": p.budget}
            if p.pattern is not None:
                row["pattern"] = list(p.pattern)
            if p.notes:
                row["notes"] = p.notes
            phrases.append(row)

    raw = {
        "song": {
            "id": parsed.song_id,
            "title": parsed.title,
            "artist": "",
            "source_lang": "ja",
        },
        "sections": sections,
        "phrases": phrases,
    }
    return Song.from_json(raw, where=f"import:{parsed.song_id}")


def _quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _plain(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.-]+", value) and value.lower() not in _YAML_WORDS:
        return value
    return _quote(value)


def dump_song(song: Song) -> str:
    """按仓库里 ``kotozute/song.yaml`` 的紧凑风格输出。"""

    out: list[str] = [
        "song:",
        f"  id: {_plain(song.id)}",
        f"  title: {_quote(song.title)}",
        '  artist: ""          # TODO 人工补全',
        f"  source_lang: {_plain(song.source_lang)}",
        "",
        "sections:",
    ]
    for section in song.sections:
        role = _plain(section.role) if section.role else '""'
        out.append(f"  - {{id: {_plain(section.id)}, role: {role}}}")

    out.extend(["", "phrases:"])
    prev_section: str | None = None
    for p in song.phrases:
        if prev_section is not None and p.section != prev_section:
            out.append("")
        prev_section = p.section

        fields = [
            f"id: {_plain(p.id)}",
            f"section: {_plain(p.section)}",
            f"ja: {_quote(p.ja)}",
        ]
        if p.keep_source:
            fields.append("keep_source: true")
        else:
            if p.budget is None:
                raise LyricsImportError(f"内部错误：{p.id} 没有 budget")
            fields.append(f"budget: {{min: {p.budget.min}, max: {p.budget.max}}}")
        if p.pattern is not None:
            fields.append("pattern: [" + ", ".join(str(x) for x in p.pattern) + "]")
        if p.notes:
            fields.append(f"notes: {_quote(p.notes)}")
        out.append("  - {" + ", ".join(fields) + "}")

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# 命令行
# --------------------------------------------------------------------------


def _total_budget(song: Song) -> int:
    return sum(p.budget.max for p in song.phrases if p.budget is not None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="import_lyrics",
        description="把「歌词 总字数 [分句...]」格式的标注文本转换成 data/songs/<id>/song.yaml",
    )
    parser.add_argument("path", type=Path, help="标注文本路径")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/songs"),
        help="歌库目录（默认 data/songs）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已经存在的 song.yaml；默认拒绝覆盖",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只解析并打印结果，不写文件",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="分句合计与总字数不一致时直接报错（默认修复或忽略并警告）",
    )
    args = parser.parse_args(argv)

    try:
        text = args.path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"读不了 {args.path}：{exc}", file=sys.stderr)
        return 2

    try:
        parsed = parse_lyrics(text, strict=args.strict)
    except LyricsImportError as exc:
        print(f"导入失败：{exc}", file=sys.stderr)
        return 2

    built: list[tuple[ParsedSong, Song]] = []
    for ps in parsed:
        try:
            song = build_song(ps)
        except (LyricsImportError, SchemaError) as exc:
            print(f"{ps.title}：{exc}", file=sys.stderr)
            return 2
        built.append((ps, song))

    print(f"{args.path}：{len(built)} 首歌")
    for ps, song in built:
        keep = sum(1 for p in song.phrases if p.keep_source)
        extra = f" / 保留原文 {keep} 句" if keep else ""
        print(
            f"  {song.title} -> {song.id}  "
            f"{len(song.sections)} 段 / {len(song.phrases)} 句{extra} / 预算 {_total_budget(song)} 字"
        )
        for w in ps.warnings:
            print(f"    ! {w}")

    if args.dry_run:
        print("\n--dry-run：没有写任何文件")
        return 0

    written = 0
    skipped = 0
    for ps, song in built:
        path = song_path(song.id, root=args.root)
        if path.exists() and not args.force:
            print(f"\n已存在，跳过（加 --force 覆盖）：{path}")
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dump_song(song), encoding="utf-8")
        print(f"\n已写出：{path}")
        written += 1

    print(f"\n完成：写入 {written} 首，跳过 {skipped} 首")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
