"""查看：每个 agent 到底产出了什么。

一个 run 目录里每个 agent 只有 ``request/output/meta`` 三个文件，直接 cat
``output.json`` 很难读，所以这里给每类产物一个面向人的渲染：

- A：体会（主题 / 情绪走向 / 质地 / 意象池 / 分段情绪 / 叙事）
- B：某一段的译词（原文 / 译词 / 字数 / 预算 / 自查说明）
- C：检查结论（verdict / 违规 / 改动 / 回归）
"""

from __future__ import annotations

import unicodedata
from typing import Any

from .counting import count_syllables
from .models import Analysis, Draft, Review, Song
from .store import RunStore, section_index
from .view import load_draft, load_review

__all__ = [
    "text_width",
    "pad",
    "table",
    "list_runs",
    "render_analysis",
    "render_draft",
    "render_review",
    "render_record",
]


def text_width(text: str) -> int:
    """中日文按两格宽算，表格才对得齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    return text + " " * max(0, width - text_width(text))


def table(rows: list[list[str]], *, gap: str = "  ") -> str:
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    widths = [max((text_width(r[i]) if i < len(r) else 0) for r in rows) for i in range(cols)]
    return "\n".join(
        gap.join(pad(r[i] if i < len(r) else "", widths[i]) for i in range(cols)).rstrip()
        for r in rows
    )


# --------------------------------------------------------------------------
# run 列表
# --------------------------------------------------------------------------


def _run_model(store: RunStore, run_id: str, song: Song) -> str:
    for agent, index in [("A", None), *[("B", i) for i in range(1, len(song.sections) + 1)]]:
        meta = store.read_meta(run_id, agent, index)  # type: ignore[arg-type]
        if meta.get("model"):
            return str(meta["model"])
    return "-"


def list_runs(store: RunStore, song: Song) -> list[dict[str, Any]]:
    """按时间顺序列出所有 run；标出 ``latest.json`` 指的是哪一个。"""
    latest = store.read_latest()
    current = latest.get("run")
    rows: list[dict[str, Any]] = []
    for rid in store.list_runs():
        passed = 0
        total = 0
        for sid in song.section_ids:
            if not song.translatable_of(sid):
                continue
            total += 1
            review = load_review(store, rid, section_index(song, sid))
            if review is not None and review.verdict == "pass":
                passed += 1
        meta = store.read_meta(rid, "A") if store.has_output(rid, "A") else {}
        usage = meta.get("usage") if isinstance(meta, dict) else None
        rows.append(
            {
                "run_id": rid,
                "current": rid == current,
                "model": _run_model(store, rid, song),
                "has_a": store.has_output(rid, "A"),
                "passed": passed,
                "total": total,
                "status": (latest.get("status") if rid == current else None) or "",
                "lyrics": (store.run_dir(rid) / "lyrics.zh.txt").is_file(),
                "out_tokens": (usage or {}).get("output_tokens") if isinstance(usage, dict) else None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------


def render_analysis(run_id: str, analysis: Analysis) -> str:
    out = [f"A 的体会　（run {run_id}）", ""]
    out.append(f"主题　　：{analysis.theme}")
    out.append(f"情绪走向：{analysis.emotional_arc}")
    if analysis.tone_words:
        out.append(f"质地　　：{' / '.join(analysis.tone_words)}")
    out.append("")

    if analysis.imagery_pool:
        out.append("意象池：")
        for item in analysis.imagery_pool:
            tail = f"　—— {item.why}" if item.why else ""
            out.append(f"  · {item.image}{tail}")
        out.append("")

    if analysis.sections:
        out.append("分段情绪：")
        for sid, mood in analysis.sections.items():
            out.append(f"  {sid}：{mood}")
        out.append("")

    out.append("叙事：")
    for line in analysis.narrative.splitlines():
        out.append(f"  {line}")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# B
# --------------------------------------------------------------------------


def render_draft(run_id: str, song: Song, draft: Draft) -> str:
    out = [f"B 的译词　run {run_id}　section {draft.section}", ""]
    by_id = draft.by_id()
    rows = [["id", "原文", "译词", "字数", "预算"]]
    notes: list[str] = []
    for p in song.phrases_of(draft.section):
        if p.keep_source:
            continue
        dp = by_id.get(p.id)
        zh = dp.zh if dp is not None else "（缺）"
        chars = count_syllables(zh) if dp is not None else 0
        budget = "-" if p.budget is None else f"{p.budget.min}-{p.budget.max}"
        rows.append([p.id, p.ja, zh, str(chars), budget])
        note = ((dp.self_check or {}).get("note") if dp is not None else "") or ""
        if note:
            notes.append(f"{p.id}　↳ {note}")
    out.append(table(rows))
    if notes:
        out.append("")
        out.extend(notes)
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# C
# --------------------------------------------------------------------------


def render_review(run_id: str, review: Review) -> str:
    out = [f"C 的检查　run {run_id}　section {review.section}", ""]
    out.append(f"结论：{review.verdict}")
    if review.violations:
        out.append("违规：")
        for v in review.violations:
            exp = "" if v.expected is None else f" 期望 {v.expected[0]}-{v.expected[1]}"
            act = "" if v.actual is None else f" 实际 {v.actual}"
            out.append(f"  [{v.severity}] {v.phrase_id} {v.code}{exp}{act}")
            if v.fix:
                out.append(f"        ↳ {v.fix}")
    else:
        out.append("违规：无")
    if review.changes.rewritten:
        out.append(f"改写过：{', '.join(review.changes.rewritten)}")
    if review.changes.unchanged:
        out.append(f"未动过：{', '.join(review.changes.unchanged)}")
    if review.regressions:
        out.append(f"回归（上一轮合格、本轮不合格）：{', '.join(review.regressions)}")
    return "\n".join(out).rstrip() + "\n"


def render_record(song: Song, store: RunStore, run_id: str, agent: str, section: str | None) -> str:
    """按 agent 选择渲染方式；``section`` 只对 B / C 有意义。"""
    agent = agent.lower()
    if agent == "a":
        payload = store.load_output(run_id, "A")
        if payload is None:
            return f"{run_id}：A 还没有产物\n"
        return render_analysis(run_id, Analysis.from_json(payload))
    if section is None:
        if not song.translatable_of(song.section_ids[0]):
            section = next((s for s in song.section_ids if song.translatable_of(s)), None)
        else:
            section = song.section_ids[0]
    if section is None:
        return f"{run_id}：这首歌没有可译的 section\n"
    index = section_index(song, section)
    if agent == "b":
        draft = load_draft(store, run_id, index)
        return f"{run_id}：{section} 还没有 B 产物\n" if draft is None else render_draft(run_id, song, draft)
    if agent == "c":
        review = load_review(store, run_id, index)
        return f"{run_id}：{section} 还没有 C 产物\n" if review is None else render_review(run_id, review)
    return f"未知 agent {agent!r}\n"
