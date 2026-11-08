"""视图装配：由 ``song.yaml`` + **一个 run 目录**拼出“当前全曲长什么样”。

一个 run 自带全部产物，装配只需要 ``song`` + ``run_id`` + ``pipeline``，不需要额外的指针文件。
视图是导出的结果，不是存储的真相；删掉它随时可以重算。

「最终文本」在哪个目录，取决于管线：

- ``workflow``：``writer``（写词的人，被 reviewer 检查的就是它）；
- ``rag``：``mimic_write``（mimic 的写作部分，检索在 ``mimic_lookup``）。

reviewer 两条管线都叫 ``reviewer``。
"""

from __future__ import annotations

from .models import Draft, Review, Song, View, ViewRow
from .store import RunStore, section_index

__all__ = [
    "PIPELINES",
    "DRAFT_AGENT",
    "AGENTS_BY_PIPELINE",
    "draft_agent",
    "build_view",
    "load_draft",
    "load_review",
    "run_section_verdicts",
]

PIPELINES = ("workflow", "rag")
#: 每个管线里"最终文本"所在的 agent 目录。
DRAFT_AGENT = {"workflow": "writer", "rag": "mimic_write"}
REVIEW_AGENT = "reviewer"
#: 每条管线在 ``sections/NNN/`` 下会出现哪些 agent 目录（抄 run 时要整份抄）。
AGENTS_BY_PIPELINE: dict[str, tuple[str, ...]] = {
    "workflow": ("writer", REVIEW_AGENT),
    "rag": ("writer", "mimic_lookup", "mimic_write", REVIEW_AGENT),
}


def draft_agent(pipeline: str) -> str:
    try:
        return DRAFT_AGENT[pipeline]
    except KeyError:
        raise KeyError(f"未知管线 {pipeline!r}，可选 {PIPELINES}") from None


def load_draft(
    store: RunStore, run_id: str | None, index: int, *, pipeline: str = "workflow"
) -> Draft | None:
    payload = store.load_output(run_id, draft_agent(pipeline), index)
    return None if payload is None else Draft.from_json(payload)


def load_review(store: RunStore, run_id: str | None, index: int) -> Review | None:
    payload = store.load_output(run_id, REVIEW_AGENT, index)
    return None if payload is None else Review.from_json(payload)


def run_section_verdicts(
    song: Song, store: RunStore, run_id: str | None
) -> dict[str, str]:
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


def build_view(
    song: Song, store: RunStore, run_id: str | None, *, pipeline: str = "workflow"
) -> View:
    drafts: dict[str, Draft | None] = {}
    reviews: dict[str, Review | None] = {}
    for sid in song.section_ids:
        idx = section_index(song, sid)
        drafts[sid] = load_draft(store, run_id, idx, pipeline=pipeline)
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
