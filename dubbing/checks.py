"""C：纯规则检查。不看语义，只数字数与结构。

一次检查 = 一个 section。除了本轮违规，还要算出：

- ``changes``：哪些句子被改过（含原本合格的）——只供人看，不算违规。
- ``regressions``：上一轮合格、本轮不合格——返修把好的改坏了。

``regressions`` 不引入新规则：它本身就是"现在坏了"，自然计入 ``verdict``。
单独列出来只是为了让人一眼看到"是改坏的，不是原本就坏的"。
"""

from __future__ import annotations

from collections import defaultdict

from .counting import contains_kana, count_syllables, has_punct
from .models import Changes, Draft, Phrase, Review, Song, Violation

__all__ = ["review"]


def _norm(text: str) -> str:
    """比较重复用时忽略空白（`像要沉下去 又像要融化` 的分隔方式不固定）。"""
    return "".join(text.split())


def review(
    song: Song,
    section_id: str,
    draft: Draft,
    *,
    draft_run: str,
    prev_draft: Draft | None = None,
    prev_review: Review | None = None,
) -> Review:
    phrases = song.phrases_of(section_id)
    translatable: list[Phrase] = [p for p in phrases if p.translatable]
    by_id = draft.by_id()

    found: dict[str, list[Violation]] = defaultdict(list)

    def add(v: Violation) -> None:
        found[v.phrase_id].append(v)

    for p in translatable:
        dp = by_id.get(p.id)
        zh = (dp.zh if dp is not None else "") or ""
        if not zh.strip():
            add(
                Violation(
                    phrase_id=p.id,
                    code="EMPTY",
                    severity="hard",
                    expected=p.budget.as_tuple() if p.budget else None,
                    actual=0,
                    fix="补一句译词",
                )
            )
            continue

        n = count_syllables(zh)
        lo_hi = p.budget.as_tuple() if p.budget is not None else None

        if contains_kana(zh):
            add(
                Violation(
                    phrase_id=p.id,
                    code="NON_ZH",
                    severity="hard",
                    expected=lo_hi,
                    actual=n,
                    fix="把残留的假名换成中文",
                )
            )

        if lo_hi is not None:
            lo, hi = lo_hi
            if n > hi:
                add(
                    Violation(
                        phrase_id=p.id,
                        code="OVER_BUDGET",
                        severity="hard",
                        expected=lo_hi,
                        actual=n,
                        fix=f"超出 {n - hi} 字，压到 {hi} 字以内",
                    )
                )
            elif n < lo:
                add(
                    Violation(
                        phrase_id=p.id,
                        code="UNDER_BUDGET",
                        severity="hard",
                        expected=lo_hi,
                        actual=n,
                        fix=f"差 {lo - n} 字，补到 {lo} 字以上",
                    )
                )

        if has_punct(zh):
            add(
                Violation(
                    phrase_id=p.id,
                    code="PUNCT",
                    severity="soft",
                    expected=lo_hi,
                    actual=n,
                    fix="去掉标点；停顿用空格表达",
                )
            )

        if dp is not None and dp.chars is not None and dp.chars != n:
            add(
                Violation(
                    phrase_id=p.id,
                    code="CHAR_MISMATCH",
                    severity="soft",
                    actual=n,
                    fix=f"自报 {dp.chars} 字，实际 {n} 字",
                )
            )

    # 原文不同却给出完全相同的译词——多半是模型偷懒
    groups: dict[str, list[str]] = defaultdict(list)
    for p in translatable:
        dp = by_id.get(p.id)
        if dp is None or not dp.zh.strip():
            continue
        groups[_norm(dp.zh)].append(p.id)
    for _, pids in groups.items():
        if len(pids) < 2:
            continue
        if len({song.phrase(pid).ja for pid in pids}) < 2:
            continue  # 原文本来就一样（副歌重复），译词相同是对的
        for pid in pids:
            others = [x for x in pids if x != pid]
            add(
                Violation(
                    phrase_id=pid,
                    code="DUPLICATE",
                    severity="hard",
                    fix=f"与 {others[0]} 原文不同却给出相同译词，换一种说法",
                )
            )

    ordered = [
        v
        for p in translatable
        for v in sorted(
            found.get(p.id, []),
            key=lambda x: (x.severity != "hard", x.code),
        )
    ]

    # -- 本轮相对上一版的变化 -------------------------------------------
    prev_texts = prev_draft.texts() if prev_draft is not None else {}
    rewritten: list[str] = []
    unchanged: list[str] = []
    for p in translatable:
        dp = by_id.get(p.id)
        cur = (dp.zh if dp is not None else "") or ""
        if p.id in prev_texts and prev_texts[p.id].strip() == cur.strip():
            unchanged.append(p.id)
        else:
            rewritten.append(p.id)

    # -- 返修把合格的改坏了 ---------------------------------------------
    regressions: tuple[str, ...] = ()
    if prev_review is not None and prev_review.section == section_id:
        now_failing = {
            pid for pid, vs in found.items() if any(v.severity == "hard" for v in vs)
        }
        was_failing = prev_review.failing_ids()
        regressions = tuple(
            p.id for p in translatable if p.id in now_failing and p.id not in was_failing
        )

    verdict = "fail" if any(v.severity == "hard" for v in ordered) else "pass"
    return Review(
        section=section_id,
        draft_run=draft_run,
        verdict=verdict,
        violations=tuple(ordered),
        changes=Changes(rewritten=tuple(rewritten), unchanged=tuple(unchanged)),
        regressions=regressions,
    )
