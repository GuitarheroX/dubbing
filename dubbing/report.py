"""查看：每个 agent 到底产出了什么。

三份产物各有自己的形状，直接 cat ``output.json`` 很难读，所以这里给每类产物
一个面向人的渲染：

- A：体会（主题 / 情绪走向 / 质地 / 意象池 / 分段情绪 / 叙事）
- B：某一段的译词（原文 / 译词 / 字数 / 预算 / 自查说明）
- C：检查结论（verdict / 违规 / 改动 / 回归）
"""

from __future__ import annotations

import unicodedata
from typing import Any

from .counting import count_syllables
from .models import Analysis, Draft, Review, Song, State
from .store import SongStore

__all__ = [
    "text_width",
    "pad",
    "table",
    "current_run_ids",
    "list_runs",
    "render_analysis",
    "render_draft",
    "render_review",
    "render_run",
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


def _meta_or_empty(store: SongStore, run_id: str) -> dict[str, Any]:
    """进程被中断的 run 可能只有 request.json，没有 meta。"""
    try:
        return store.load_meta(run_id)
    except FileNotFoundError:
        return {}


def current_run_ids(state: State) -> set[str]:
    ids = set()
    if state.analysis:
        ids.add(state.analysis)
    for ss in state.sections.values():
        for rid in (ss.draft, ss.review):
            if rid:
                ids.add(rid)
    return ids


def list_runs(store: SongStore, state: State) -> list[dict[str, Any]]:
    """按时间顺序列出所有 run，并标出当前视图用的是哪几个。"""
    current = current_run_ids(state)
    rows: list[dict[str, Any]] = []
    for rid in store.run_ids():
        meta = _meta_or_empty(store, rid)
        out: dict[str, Any] = {}
        if store.has_run(rid):
            out = store.load_output(rid)

        agent = meta.get("agent") or SongStore._agent_of(rid) or "?"
        if agent == "a":
            note = f"{len(out.get('sections', {}))} 段"
        elif agent == "b":
            note = f"{len(out.get('phrases', []))} 句"
        elif agent == "c":
            note = out.get("verdict", "?")
            if out.get("violations"):
                note += f"（{len(out['violations'])} 条）"
        else:
            note = ""

        if meta.get("abandoned"):
            note = f"未完成：{str(meta['abandoned'])[:40]}"
        elif not meta:
            note = "未完成（进程中断）"

        usage = meta.get("usage") or {}
        rows.append(
            {
                "run_id": rid,
                "agent": agent,
                "model": meta.get("model") or "-",
                "round": meta.get("round"),
                "out_tokens": usage.get("output_tokens"),
                "note": note,
                "current": rid in current,
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
    out.append(f"结论：{review.verdict}　（draft {review.draft_run}）")
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


def render_run(store: SongStore, song: Song, run_id: str) -> str:
    """按 run 的 agent 自动选择渲染方式。"""
    meta = _meta_or_empty(store, run_id)
    if not meta:
        return f"{run_id}：未完成（没有 meta.json，可能进程中断）\n"
    if meta.get("abandoned"):
        return f"{run_id}（{meta.get('agent')}）：未完成——{meta['abandoned']}\n"
    out = store.load_output(run_id)
    agent = meta.get("agent")
    if agent == "a":
        return render_analysis(run_id, Analysis.from_json(out))
    if agent == "b":
        return render_draft(run_id, song, Draft.from_json(out))
    if agent == "c":
        return render_review(run_id, Review.from_json(out))
    return f"{run_id}：未知 agent {agent!r}\n"
