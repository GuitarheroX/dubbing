"""导出：把视图变成能直接看、直接唱的歌词文件。

两种产物：

- ``lyrics.zh.txt``：纯中文歌词，按段落分开。给演唱、复制、发给别人看。
- ``lyrics.md``：原文 / 译词 / 字数对照表。给自己校对"意思有没有跑"。
"""

from __future__ import annotations

from pathlib import Path

from .counting import count_syllables
from .models import Song, View

__all__ = ["zh_lyrics", "side_by_side", "write_lyrics", "ZH_NAME", "MD_NAME"]

ZH_NAME = "lyrics.zh.txt"
MD_NAME = "lyrics.md"


def _budget_label(row_budget) -> str:
    if row_budget is None:
        return "-"
    if row_budget.min == row_budget.max:
        return str(row_budget.min)
    return f"{row_budget.min}-{row_budget.max}"


def zh_lyrics(song: Song, view: View) -> str:
    out: list[str] = [song.title, ""]
    for sec in song.sections:
        rows = view.of_section(sec.id)
        out.append(f"[{sec.id}]")
        for r in rows:
            out.append(r.sing_text or f"〈未译〉{r.ja}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def side_by_side(song: Song, view: View) -> str:
    done = sum(1 for r in view.rows if r.verdict in {"pass", "skip"})
    out: list[str] = [
        f"# {song.title}",
        "",
        f"- 乐句：{len(view.rows)} 句，已定稿 {done} 句",
        f"- 数据：data/songs/{song.id}/song.yaml",
        "",
    ]
    for sec in song.sections:
        out.append(f"## {sec.id}　{sec.role}".rstrip())
        out.append("")
        out.append("| 原文 | 译词 | 字数 / 预算 | 状态 |")
        out.append("| --- | --- | --- | --- |")
        for r in view.of_section(sec.id):
            chars = "—" if r.lang != "zh" else str(count_syllables(r.sing_text))
            out.append(
                f"| {r.ja} | {r.sing_text or '—'} | {chars} / {_budget_label(r.budget)} | {r.verdict} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_lyrics(song: Song, view: View, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    zh = out_dir / ZH_NAME
    md = out_dir / MD_NAME
    zh.write_text(zh_lyrics(song, view), encoding="utf-8")
    md.write_text(side_by_side(song, view), encoding="utf-8")
    return [zh, md]
