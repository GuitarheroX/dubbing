"""编排：A 一次 → 每个 section 循环（B → C → 不合格就带着 prev_zh 重来）。

结构就是 README 里那条 workflow。没有引入 LangGraph：节点与边的形状已经
一一对应（``ensure_analysis`` / ``attempt_section`` / 循环边），换成 LangGraph
时把这里的状态换成它的 State、把这两个函数换成节点即可。

状态一律落盘：``state.yaml`` 存指针，产物在 ``runs/``。所以中断后重跑
会从上次的 draft 继续，而不是从零开始。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import checks
from .agents import build_a_request, build_b_request, parse_a_output, parse_b_output
from .llm import LLMClient, LLMError, NeedsLLM, extract_json
from .models import Analysis, Draft, Review, Song, State
from .prompting import prompt_hash
from .song import load_song
from .store import Request, Run, SongStore
from .view import build_view, load_draft, load_review

__all__ = [
    "Step",
    "SongOutcome",
    "SectionOutcome",
    "run_song",
    "plan_requests",
]

DEFAULT_ROOT = Path("data/songs")
DEFAULT_MAX_ROUNDS = 3


@dataclass(frozen=True)
class Step:
    """编排器走到哪一步了。``phase`` 取 ``start`` / ``done``。"""

    agent: str  # "a" | "b" | "c"
    phase: str
    section: str | None = None
    note: str = ""


StepHook = Callable[[Step], None]


@dataclass
class SectionOutcome:
    section: str
    status: str  # "passed" | "exhausted" | "skipped" | "reused"
    rounds: int = 0
    draft_run: str | None = None
    review_run: str | None = None
    hard: int = 0
    soft: int = 0
    regressions: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)


@dataclass
class SongOutcome:
    song_id: str
    analysis_run: str | None = None
    analysis_reused: bool = False
    sections: list[SectionOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.status in {"passed", "reused", "skipped"} for s in self.sections)


def _meta(agent: str, llm: LLMClient, system: str, **extra: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "agent": agent,
        "model": getattr(llm, "model", llm.name),
        "client": llm.name,
        "prompt_hash": prompt_hash(system),
        **extra,
    }
    usage = getattr(llm, "last_usage", None)
    if usage:
        meta["usage"] = usage
    temperature = getattr(llm, "last_temperature", None)
    if temperature is not None:
        meta["temperature"] = temperature
    return meta


def _call(
    store: SongStore, run: Run, llm: LLMClient, *, agent: str, request: Request
) -> str:
    try:
        return llm.complete(agent=agent, system=request.system, user=request.user)
    except NeedsLLM as exc:
        store.abandon(run, reason="needs-llm")
        raise NeedsLLM(f"{exc}\n本次入参已落盘：{run.request_path}") from exc
    except LLMError as exc:
        store.abandon(run, reason=f"error: {exc}")
        raise LLMError(f"{exc}\n本次入参已落盘：{run.request_path}") from exc


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------


def ensure_analysis(
    song: Song,
    store: SongStore,
    state: State,
    llm: LLMClient,
    *,
    force: bool = False,
    on_step: StepHook | None = None,
) -> tuple[Analysis, str, bool, list[str]]:
    """返回 ``(analysis, run_id, reused, warnings)``。"""

    def emit(step: Step) -> None:
        if on_step is not None:
            on_step(step)

    if not force and store.has_run(state.analysis):
        analysis = Analysis.from_json(store.load_output(state.analysis))  # type: ignore[arg-type]
        missing = [sid for sid in song.section_ids if sid not in analysis.sections]
        warnings: list[str] = []
        if missing:
            warnings.append(
                f"A 的产物没有覆盖 {missing}（song.yaml 改过？）——"
                "这些段会缺少情绪参考，建议 --redo-analysis"
            )
            emit(Step("a", "done", note=f"复用上一版，缺 {'/'.join(missing)}"))
        else:
            emit(Step("a", "done", note="复用上一版"))
        return analysis, state.analysis, True, warnings  # type: ignore[return-value]

    emit(Step("a", "start", note="体会整首歌"))
    request = build_a_request(song)
    run = store.new_run("a", request)
    raw = _call(store, run, llm, agent="a", request=request)
    analysis, warnings = parse_a_output(extract_json(raw), song)
    store.finish(run, analysis.to_json(), meta=_meta("a", llm, request.system, warnings=warnings))
    state.analysis = run.id
    store.save_state(state)
    emit(Step("a", "done", note=f"{len(analysis.sections)} 段"))
    return analysis, run.id, False, warnings


# --------------------------------------------------------------------------
# B + C
# --------------------------------------------------------------------------


def _accepted_texts(song: Song, state: State, store: SongStore, focus: str) -> dict[str, str]:
    view = build_view(song, state, store)
    return view.accepted_texts(exclude={p.id for p in song.phrases_of(focus)})


def _review_request(song: Song, section: str, draft: Draft, prev: Draft | None) -> Request:
    data = {
        "section": section,
        "phrases": [p.to_json() for p in song.phrases_of(section)],
        "draft": draft.to_json(),
        "prev_draft": prev.to_json() if prev is not None else None,
    }
    return Request(system="(C 是规则检查，不调用模型)", user="", data=data)


def attempt_section(
    song: Song,
    store: SongStore,
    state: State,
    analysis: Analysis,
    llm: LLMClient,
    section: str,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    skip_done: bool = False,
    fresh: bool = True,
    on_step: StepHook | None = None,
) -> SectionOutcome:
    ss = state.of(section)

    def emit(step: Step) -> None:
        if on_step is not None:
            on_step(step)

    if not song.translatable_of(section):
        ss.verdict = "skipped"
        store.save_state(state)
        emit(Step("b", "done", section=section, note="整段保留原文，跳过"))
        return SectionOutcome(section=section, status="skipped")

    if skip_done and ss.verdict == "pass" and store.has_run(ss.draft):
        emit(Step("b", "done", section=section, note="已完成，跳过"))
        return SectionOutcome(
            section=section,
            status="reused",
            draft_run=ss.draft,
            review_run=ss.review,
        )

    # fresh：忽略这一段已有的译文，从零开始（旧 run 仍在磁盘上，不会被删）
    prev_draft: Draft | None = None if fresh else load_draft(store, ss.draft)
    prev_review: Review | None = None if fresh else load_review(store, ss.review)

    outcome = SectionOutcome(section=section, status="exhausted")
    for attempt in range(1, max_rounds + 1):
        accepted = _accepted_texts(song, state, store, section)
        prev_texts = prev_draft.texts() if prev_draft is not None else None
        request = build_b_request(
            song,
            analysis,
            focus=section,
            accepted=accepted,
            prev_texts=prev_texts,
            feedback=prev_review.violations if prev_review is not None else (),
        )
        emit(Step("b", "start", section=section, note=f"第 {attempt} 轮"))
        run = store.new_run("b", request)
        raw = _call(store, run, llm, agent="b", request=request)
        draft, warnings = parse_b_output(extract_json(raw), song, section)
        store.finish(
            run,
            draft.to_json(),
            meta=_meta("b", llm, request.system, round=attempt, warnings=warnings),
        )

        review = checks.review(
            song,
            section,
            draft,
            draft_run=run.id,
            prev_draft=prev_draft,
            prev_review=prev_review,
        )
        review_run = store.new_run("c", _review_request(song, section, draft, prev_draft))
        store.finish(review_run, review.to_json(), meta={"agent": "c", "rule_based": True})

        ss.draft, ss.review, ss.round, ss.verdict = run.id, review_run.id, attempt, review.verdict
        store.save_state(state)
        soft = len(review.violations) - len(review.hard)
        emit(
            Step(
                "c",
                "done",
                section=section,
                note=f"{review.verdict}（{len(review.hard)} 硬 / {soft} 软）",
            )
        )

        outcome = SectionOutcome(
            section=section,
            status="passed" if review.verdict == "pass" else "exhausted",
            rounds=attempt,
            draft_run=run.id,
            review_run=review_run.id,
            hard=len(review.hard),
            soft=len(review.violations) - len(review.hard),
            regressions=review.regressions,
            warnings=warnings,
        )
        if review.verdict == "pass":
            return outcome

        prev_draft, prev_review = draft, review

    return outcome


# --------------------------------------------------------------------------
# 整首歌
# --------------------------------------------------------------------------


def run_song(
    song_id: str,
    llm: LLMClient,
    *,
    root: Path | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    skip_done: bool = False,
    fresh: bool = True,
    redo_analysis: bool = False,
    only: set[str] | None = None,
    on_step: StepHook | None = None,
) -> SongOutcome:
    song = load_song(song_id, root=root)
    if only:
        unknown = sorted(only - set(song.section_ids))
        if unknown:
            raise KeyError(f"没有这些 section：{unknown}，可选 {list(song.section_ids)}")
    store = SongStore(root or DEFAULT_ROOT, song_id)
    state = store.load_state()

    analysis, analysis_run, reused, warnings = ensure_analysis(
        song, store, state, llm, force=redo_analysis, on_step=on_step
    )
    outcome = SongOutcome(song_id=song_id, analysis_run=analysis_run, analysis_reused=reused)
    outcome.warnings.extend(warnings)

    targets = [s for s in song.sections if only is None or s.id in only]
    for section in targets:
        outcome.sections.append(
            attempt_section(
                song,
                store,
                state,
                analysis,
                llm,
                section.id,
                max_rounds=max_rounds,
                skip_done=skip_done,
                fresh=fresh,
                on_step=on_step,
            )
        )
    store.save_state(state)
    return outcome


def plan_requests(
    song_id: str,
    *,
    root: Path | None = None,
    skip_done: bool = False,
    fresh: bool = True,
) -> list[tuple[str, Request]]:
    """渲染出所有待发请求但不调用模型（给 ``--dry-run`` 用）。"""
    song = load_song(song_id, root=root)
    store = SongStore(root or DEFAULT_ROOT, song_id)
    state = store.load_state()

    plan: list[tuple[str, Request]] = []
    if not store.has_run(state.analysis):
        plan.append(("a", build_a_request(song)))
        # 没有 A 的产物时，B 的 analysis 只能留空
        analysis = Analysis(theme="", emotional_arc="", narrative="")
    else:
        analysis = Analysis.from_json(store.load_output(state.analysis))  # type: ignore[arg-type]

    for section in song.sections:
        ss = state.sections.get(section.id)
        if not song.translatable_of(section.id):
            continue
        if skip_done and ss is not None and ss.verdict == "pass" and store.has_run(ss.draft):
            continue
        prev = None if fresh else load_draft(store, ss.draft if ss is not None else None)
        plan.append(
            (
                f"b:{section.id}",
                build_b_request(
                    song,
                    analysis,
                    focus=section.id,
                    accepted=_accepted_texts(song, state, store, section.id),
                    prev_texts=prev.texts() if prev is not None else None,
                ),
            )
        )
    return plan
