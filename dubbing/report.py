"""查看：每个 agent 到底产出了什么。

一个 run 目录里每个 agent 只有 ``request/output/meta`` 三个文件，直接 cat
``output.json`` 很难读，所以这里给每类产物一个面向人的渲染：

- ``perceiver``：体会（主题 / 情绪走向 / 质地 / 意象池 / 分段情绪 / 叙事）
- ``writer`` / ``mimic_write``：某一段的译词（原文 / 译词 / 字数 / 预算 / 自查说明）
- ``mimic_lookup``：这一段拿什么去检索、检索到什么
- ``reviewer``：检查结论（verdict / 违规 / 改动 / 回归）
"""

from __future__ import annotations

import unicodedata
from typing import Any

from .counting import count_syllables
from .models import Analysis, Draft, LookupResult, Review, Song
from .store import RunStore, section_index
from .view import REVIEW_AGENT, load_review

__all__ = [
    "text_width",
    "pad",
    "table",
    "list_runs",
    "render_analysis",
    "render_draft",
    "render_lookup",
    "render_review",
    "render_record",
]

AGENT_LABELS = {
    "perceiver": "perceiver",
    "writer": "writer",
    "mimic_write": "mimic（写作）",
    "mimic_lookup": "mimic（检索）",
    "reviewer": "reviewer",
}
#: ``show`` 里不带 ``--agent`` 时按顺序打印哪些。
ALL_AGENTS = ("perceiver", "writer", "mimic_lookup", "mimic_write", "reviewer")


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
    """第一个记了 model 的 agent 就是这一轮的模型（两条管线的 agent 名都试一遍）。"""
    candidates: list[tuple[str, int | None]] = [("perceiver", None)]
    for index in range(1, len(song.sections) + 1):
        candidates.extend(
            (agent, index) for agent in ("writer", "mimic_write", "mimic_lookup")
        )
    for agent, index in candidates:
        meta = store.read_meta(run_id, agent, index)
        if meta.get("model"):
            return str(meta["model"])
    return "-"


def list_runs(store: RunStore, song: Song) -> list[dict[str, Any]]:
    """按时间顺序列出所有 run；标出 ``latest.json`` 指的是哪一个。

    两条管线的 run 目录是分开的（``workflow/`` 与 ``workflow_rag/``），所以这里不需要
    ``pipeline`` 参数——直接看传进来的那个 store。
    """
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
        meta = store.read_meta(rid, "perceiver") if store.has_output(rid, "perceiver") else {}
        usage = meta.get("usage") if isinstance(meta, dict) else None
        rows.append(
            {
                "run_id": rid,
                "current": rid == current,
                "model": _run_model(store, rid, song),
                "has_perceiver": store.has_output(rid, "perceiver"),
                "passed": passed,
                "total": total,
                "status": (latest.get("status") if rid == current else None) or "",
                "lyrics": (store.run_dir(rid) / "lyrics.zh.txt").is_file(),
                "out_tokens": (usage or {}).get("output_tokens") if isinstance(usage, dict) else None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# perceiver
# --------------------------------------------------------------------------


def render_analysis(run_id: str, analysis: Analysis) -> str:
    out = [f"perceiver 的体会　（run {run_id}）", ""]
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
# writer / mimic_write
# --------------------------------------------------------------------------


def render_draft(run_id: str, song: Song, draft: Draft, *, agent: str = "writer") -> str:
    label = AGENT_LABELS.get(agent, agent)
    out = [f"{label} 的译词　run {run_id}　section {draft.section}", ""]
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
# mimic_lookup
# --------------------------------------------------------------------------


def render_lookup(run_id: str, result: LookupResult) -> str:
    out = [f"mimic 的检索　run {run_id}　section {result.section}", ""]
    out.append("查询（本段初版译词）：")
    lines = result.query_text.splitlines() or ["（空）"]
    out.extend(f"  {line}" for line in lines)
    out.append("")
    out.append(
        f"索引：{result.index_id or '—'}　collection {result.collection or '—'}　k={result.k}"
    )
    if result.skipped:
        out.append(f"跳过：{result.skipped}")
    if not result.hits:
        out.append("命中：无")
        return "\n".join(out).rstrip() + "\n"
    out.append("命中：")
    for i, hit in enumerate(result.hits, 1):
        who = f"　{hit.poet}" if hit.poet else ""
        title = f"《{hit.title}》" if hit.title else "（无题）"
        out.append(f"  {i}. [{hit.score:.4f}] {title}{who}")
        for line in hit.text.splitlines():
            out.append(f"       {line}")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# reviewer
# --------------------------------------------------------------------------


def render_review(run_id: str, review: Review) -> str:
    out = [f"reviewer 的检查　run {run_id}　section {review.section}", ""]
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


def render_record(
    song: Song, store: RunStore, run_id: str, agent: str, section: str | None
) -> str:
    """按 agent 选择渲染方式；``section`` 对除 perceiver 外的 agent 都有意义。

    这里直接按 ``agent`` 名字读目录，不经过 ``view.load_draft``——因为 rag 管线里
    ``writer`` 与 ``mimic_write`` 是两份不同的产物，都需要能单独查看。
    """
    agent = agent.lower()
    if agent == "perceiver":
        payload = store.load_output(run_id, "perceiver")
        if payload is None:
            return f"{run_id}：perceiver 还没有产物\n"
        return render_analysis(run_id, Analysis.from_json(payload))

    if section is None:
        section = next((s for s in song.section_ids if song.translatable_of(s)), None)
    if section is None:
        return f"{run_id}：这首歌没有可译的 section\n"
    index = section_index(song, section)

    if agent in {"writer", "mimic_write"}:
        payload = store.load_output(run_id, agent, index)
        if payload is None:
            return f"{run_id}：{section} 还没有 {agent} 产物\n"
        return render_draft(run_id, song, Draft.from_json(payload), agent=agent)

    if agent == "mimic_lookup":
        payload = store.load_output(run_id, "mimic_lookup", index)
        if payload is None:
            return f"{run_id}：{section} 还没有 mimic_lookup 产物\n"
        return render_lookup(run_id, LookupResult.from_json(payload))

    if agent == REVIEW_AGENT:
        review = load_review(store, run_id, index)
        if review is None:
            return f"{run_id}：{section} 还没有 reviewer 产物\n"
        return render_review(run_id, review)

    return f"未知 agent {agent!r}\n"
