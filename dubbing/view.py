"""视图装配：由 ``song.yaml`` + **一个 run 目录**拼出“当前全曲长什么样”。

一个 run 自带全部产物（A 在 run 根，B/C 在 ``sections/NNN/`` 下），所以装配
只需要 ``song`` + ``run_id``，不再需要一个单独的指针文件。视图是导出的结果，
不是存储的真相；删掉它随时可以重算。
"""

from __future__ import annotations

from .models import Draft, Review, Song, View, ViewRow
from .store import RunStore, section_index

__all__ = ["build_view", "load_draft", "load_review", "run_section_verdicts"]


def load_draft(store: RunStore, run_id: str | None, index: int) -> Draft | None:
    payload = store.load_output(run_id, "B", index)
    return None if payload is None else Draft.from_json(payload)


def load_review(store: RunStore, run_id: str | None, index: int) -> Review | None:
    payload = store.load_output(run_id, "C", index)
    return None if payload is None else Review.from_json(payload)


def run_section_verdicts(song: Song, store: RunStore, run_id: str | None) -> dict[str, str]:
    """每个 section 的最终结论：``pass`` / ``fail`` / ``pending`` / ``skipped``。"""
    out: dict[str, str] = {}
    for idx, sid in enumerate(song.section_ids, 1):
        if not song.translatable_of(sid):
            out[sid] = "skipped"
            continue
        review = load_review(store, run_id, idx)
        if review is None:
            out[sid] = "pending"
        else:
            out[sid] = review.verdict
    return out


def build_view(song: Song, store: RunStore, run_id: str | None) -> View:
    drafts: dict[str, Draft | None] = {}
    reviews: dict[str, Review | None] = {}
    for sid in song.section_ids:
        idx = section_index(song, sid)
        drafts[sid] = load_draft(store, run_id, idx)
        reviews[sid] = load_review(store, run_id, idx)

    rows: list[ViewRow] = []
    for p in song.phrases:
        if p.keep_source:
            rows.append(
                ViewRow(
                    phrase_id=p.id,
                    section=p.section,
                    ja=p.ja,
                    budget=p.budget,
                    lang="ja",
                    sing_text=p.ja,
                    verdict="skip",
                )
            )
            continue

        draft = drafts.get(p.section)
        rev = reviews.get(p.section)
        text = ""
        if draft is not None:
            text = (draft.texts().get(p.id) or "").strip()

        if rev is not None:
            verdict = "fail" if p.id in rev.failing_ids() else "pass"
        else:
            verdict = "pending"

        rows.append(
            ViewRow(
                phrase_id=p.id,
                section=p.section,
                ja=p.ja,
                budget=p.budget,
                lang="zh" if text else None,
                sing_text=text,
                verdict=verdict,
                draft_run=run_id,
            )
        )

    return View(song_id=song.id, rows=tuple(rows))
