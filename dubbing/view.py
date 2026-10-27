"""视图装配：由 ``song.yaml`` + ``state.yaml`` 指向的 run 拼出"当前全曲长什么样"。

视图是导出的结果，不是存储的真相；删掉它随时可以重算。
"""

from __future__ import annotations

from .models import Draft, Review, Song, State, View, ViewRow
from .store import SongStore

__all__ = ["build_view", "load_draft", "load_review"]


def load_draft(store: SongStore, run_id: str | None) -> Draft | None:
    if not run_id or not store.has_run(run_id):
        return None
    return Draft.from_json(store.load_output(run_id))


def load_review(store: SongStore, run_id: str | None) -> Review | None:
    if not run_id or not store.has_run(run_id):
        return None
    return Review.from_json(store.load_output(run_id))


def build_view(song: Song, state: State, store: SongStore) -> View:
    rows: list[ViewRow] = []

    drafts: dict[str, Draft | None] = {}
    reviews: dict[str, Review | None] = {}
    for sid in song.section_ids:
        ss = state.sections.get(sid)
        drafts[sid] = load_draft(store, ss.draft if ss else None)
        reviews[sid] = load_review(store, ss.review if ss else None)

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

        ss = state.sections.get(p.section)
        rows.append(
            ViewRow(
                phrase_id=p.id,
                section=p.section,
                ja=p.ja,
                budget=p.budget,
                lang="zh" if text else None,
                sing_text=text,
                verdict=verdict,
                draft_run=ss.draft if ss else None,
            )
        )

    return View(song_id=song.id, rows=tuple(rows))
