"""编排：A 一次 → 每个 section 循环（B → C → 不合格就带着 prev_zh 重来）。

结构就是 README 里那条 workflow。**一次 ``run_song`` 产生一个 run 目录**：

    data/songs/<song_id>/workflow/run-<ts>-<hash>/
        A/                      # 整曲一次
        sections/001/B/         # 这一段最终一轮的译词
        sections/001/C/         # 这一段的规则检查
        lyrics.zh.txt / lyrics.md

每个 agent 目录固定 ``request.json`` / ``output.json`` / ``meta.json`` 三个文件；
多轮返修只保留**最终一轮**（返修过程仍能从该轮 request 里的反馈和 ``prev_zh`` 看到）。
``--only`` / ``--skip-done`` 会把没重跑的段落整份抄进新 run，所以每个 run 都是完整快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import checks
from .agents import build_a_request, build_b_request, parse_a_output, parse_b_output
from .export import write_lyrics
from .llm import LLMClient, LLMError, NeedsLLM, extract_json
from .models import Analysis, Draft, Review, SchemaError, Song
from .prompting import prompt_hash
from .song import load_song
from .store import Request, RunStore, run_base, section_index
from .view import build_view, load_draft, load_review

__all__ = [
    "Step",
    "compose_request",
    "SongOutcome",
    "SectionOutcome",
    "run_song",
    "plan_requests",
    "workflow_store",
]

DEFAULT_ROOT = Path("data/songs")
DEFAULT_MAX_ROUNDS = 3


def workflow_store(root: Path | str | None, song_id: str) -> RunStore:
    return RunStore(run_base(root or DEFAULT_ROOT, song_id, "workflow"))


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
    hard: int = 0
    soft: int = 0
    regressions: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)


@dataclass
class SongOutcome:
    song_id: str
    run_id: str | None = None
    analysis_reused: bool = False
    sections: list[SectionOutcome] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lyrics_paths: list[Path] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.status in {"passed", "reused", "skipped"} for s in self.sections)


def _meta(agent: str, llm: LLMClient, system: str, **extra: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "model": getattr(llm, "model", llm.name),
        "client": llm.name,
        "prompt_hash": prompt_hash(system),
        **extra,
    }
    for key, attr in (("base_url", "base_url"), ("protocol", "protocol")):
        value = getattr(llm, attr, None)
        if value is not None:
            meta[key] = value
    usage = getattr(llm, "last_usage", None)
    if usage:
        meta["usage"] = usage
    temperature = getattr(llm, "last_temperature", None)
    if temperature is not None:
        meta["temperature"] = temperature
    return meta


def _invoke(
    store: RunStore,
    run_id: str,
    agent: str,
    llm: LLMClient,
    request: Request,
    *,
    section_index_: int | None = None,
) -> str:
    """写 request.json → 调模型。失败时把原因和已落盘位置写进 meta.json。"""
    directory = store.prepare(run_id, agent, request, section_index_=section_index_)
    try:
        return llm.complete(agent=agent.lower(), system=request.system, user=request.user)
    except NeedsLLM as exc:
        store.abandon(run_id, agent, reason="needs-llm", section_index_=section_index_)
        raise NeedsLLM(f"{exc}\n本次入参已落盘：{directory / 'request.json'}") from exc
    except LLMError as exc:
        store.abandon(run_id, agent, reason=f"error: {exc}", section_index_=section_index_)
        raise LLMError(f"{exc}\n本次入参已落盘：{directory / 'request.json'}") from exc


# --------------------------------------------------------------------------
# A
# --------------------------------------------------------------------------


def ensure_analysis(
    song: Song,
    store: RunStore,
    run_id: str,
    source_run: str | None,
    llm: LLMClient,
    *,
    force: bool = False,
    on_step: StepHook | None = None,
) -> tuple[Analysis, bool, list[str]]:
    """返回 ``(analysis, reused, warnings)``。可以复用时就整份抄进新 run。"""

    def emit(step: Step) -> None:
        if on_step is not None:
            on_step(step)

    if not force and store.has_output(source_run, "A"):
        store.copy_record(source_run, run_id, "A")  # type: ignore[arg-type]
        analysis = Analysis.from_json(store.read_output(run_id, "A"))
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
        return analysis, True, warnings

    emit(Step("a", "start", note="体会整首歌"))
    request = build_a_request(song)
    raw = _invoke(store, run_id, "A", llm, request)
    try:
        analysis, warnings = parse_a_output(extract_json(raw), song)
    except (LLMError, SchemaError) as exc:
        store.abandon(run_id, "A", reason=f"解析失败：{exc}", raw_response=raw)
        raise
    store.finish(
        run_id,
        "A",
        analysis.to_json(),
        meta=_meta("a", llm, request.system, warnings=warnings),
        raw_response=raw,
    )
    emit(Step("a", "done", note=f"{len(analysis.sections)} 段"))
    return analysis, False, warnings


# --------------------------------------------------------------------------
# B + C
# --------------------------------------------------------------------------


def _accepted_texts(song: Song, store: RunStore, run_id: str, focus: str) -> dict[str, str]:
    view = build_view(song, store, run_id)
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
    store: RunStore,
    run_id: str,
    source_run: str | None,
    analysis: Analysis,
    llm: LLMClient,
    section: str,
    *,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    skip_done: bool = False,
    fresh: bool = True,
    on_step: StepHook | None = None,
) -> SectionOutcome:
    index = section_index(song, section)

    def emit(step: Step) -> None:
        if on_step is not None:
            on_step(step)

    if not song.translatable_of(section):
        emit(Step("b", "done", section=section, note="整段保留原文，跳过"))
        return SectionOutcome(section=section, status="skipped")

    if skip_done and store.has_output(source_run, "C", index):
        review = Review.from_json(store.read_output(source_run, "C", index))  # type: ignore[arg-type]
        if review.verdict == "pass":
            store.copy_record(source_run, run_id, "B", section_index_=index)  # type: ignore[arg-type]
            store.copy_record(source_run, run_id, "C", section_index_=index)  # type: ignore[arg-type]
            emit(Step("b", "done", section=section, note="已完成，跳过"))
            return SectionOutcome(section=section, status="reused")

    # fresh：忽略这一段已有的译文，从零开始（旧 run 仍在磁盘上，不会被删）
    prev_draft = None if fresh else load_draft(store, source_run, index)
    prev_review = None if fresh else load_review(store, source_run, index)

    outcome = SectionOutcome(section=section, status="exhausted")
    for attempt in range(1, max_rounds + 1):
        accepted = _accepted_texts(song, store, run_id, section)
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
        raw = _invoke(store, run_id, "B", llm, request, section_index_=index)
        try:
            draft, warnings = parse_b_output(extract_json(raw), song, section)
        except (LLMError, SchemaError) as exc:
            store.abandon(run_id, "B", reason=f"解析失败：{exc}", raw_response=raw, section_index_=index)
            raise
        store.finish(
            run_id,
            "B",
            draft.to_json(),
            meta=_meta("b", llm, request.system, round=attempt, warnings=warnings),
            raw_response=raw,
            section_index_=index,
        )

        review = checks.review(
            song,
            section,
            draft,
            draft_run=run_id,
            prev_draft=prev_draft,
            prev_review=prev_review,
        )
        store.prepare(run_id, "C", _review_request(song, section, draft, prev_draft), section_index_=index)
        store.finish(
            run_id,
            "C",
            review.to_json(),
            meta={"rule_based": True},
            section_index_=index,
        )

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
            hard=len(review.hard),
            soft=soft,
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
    store = workflow_store(root, song_id)
    source_run = store.latest_run()
    run_id = store.new_run()

    analysis, reused, warnings = ensure_analysis(
        song, store, run_id, source_run, llm, force=redo_analysis, on_step=on_step
    )
    outcome = SongOutcome(song_id=song_id, run_id=run_id, analysis_reused=reused)
    outcome.warnings.extend(warnings)

    for section in song.sections:
        index = section_index(song, section.id)
        if only is not None and section.id not in only:
            # 没被点名的段落整份抄过来，保证这个 run 仍是完整快照
            if source_run is not None:
                store.copy_record(source_run, run_id, "B", section_index_=index)
                store.copy_record(source_run, run_id, "C", section_index_=index)
            continue
        outcome.sections.append(
            attempt_section(
                song,
                store,
                run_id,
                source_run,
                analysis,
                llm,
                section.id,
                max_rounds=max_rounds,
                skip_done=skip_done,
                fresh=fresh,
                on_step=on_step,
            )
        )

    view = build_view(song, store, run_id)
    outcome.lyrics_paths = write_lyrics(song, view, store.run_dir(run_id))
    store.set_latest(
        run_id,
        song=song_id,
        status="completed" if outcome.ok else "incomplete",
        model=getattr(llm, "model", llm.name),
    )
    return outcome


def compose_request(
    song_id: str,
    *,
    agent: str = "b",
    section: str | None = None,
    root: Path | None = None,
    fresh: bool = True,
) -> Request:
    """现拼一份“如果现在要调用，会发出去什么”的请求（不调用模型）。

    跟真实调用唯一的差别：返修轮次里 C 的结论不会出现，因为这里没有“上一轮”。
    想看带反馈的原样请求，用 ``--run <id>`` 读 ``sections/NNN/B/request.json``。
    """
    song = load_song(song_id, root=root)
    store = workflow_store(root, song_id)
    source_run = store.latest_run()

    if agent == "a":
        return build_a_request(song)

    if section is None:
        section = next((s.id for s in song.sections if song.translatable_of(s.id)), None)
        if section is None:
            raise KeyError("这首歌没有可译的 section")
    if section not in song.section_ids:
        raise KeyError(f"未知 section {section!r}，可选 {list(song.section_ids)}")

    analysis = (
        Analysis.from_json(store.read_output(source_run, "A"))  # type: ignore[arg-type]
        if store.has_output(source_run, "A")
        else Analysis(theme="", emotional_arc="", narrative="")
    )
    index = section_index(song, section)
    prev = None if fresh else load_draft(store, source_run, index)
    return build_b_request(
        song,
        analysis,
        focus=section,
        accepted=_accepted_texts(song, store, source_run, section) if source_run else {},
        prev_texts=prev.texts() if prev is not None else None,
    )


def plan_requests(
    song_id: str,
    *,
    root: Path | None = None,
    skip_done: bool = False,
    fresh: bool = True,
) -> list[tuple[str, Request]]:
    """渲染出所有待发请求但不调用模型（给 ``--dry-run`` 用）。"""
    song = load_song(song_id, root=root)
    store = workflow_store(root, song_id)
    source_run = store.latest_run()

    plan: list[tuple[str, Request]] = []
    if not store.has_output(source_run, "A"):
        plan.append(("A", build_a_request(song)))
        # 没有 A 的产物时，B 的 analysis 只能留空
        analysis = Analysis(theme="", emotional_arc="", narrative="")
    else:
        analysis = Analysis.from_json(store.read_output(source_run, "A"))  # type: ignore[arg-type]

    for section in song.sections:
        if not song.translatable_of(section.id):
            continue
        index = section_index(song, section.id)
        if skip_done and store.has_output(source_run, "C", index):
            review = Review.from_json(store.read_output(source_run, "C", index))  # type: ignore[arg-type]
            if review.verdict == "pass":
                continue
        prev = None if fresh else load_draft(store, source_run, index)
        plan.append(
            (
                f"B-{section.id}",
                build_b_request(
                    song,
                    analysis,
                    focus=section.id,
                    accepted=_accepted_texts(song, store, source_run, section.id) if source_run else {},
                    prev_texts=prev.texts() if prev is not None else None,
                ),
            )
        )
    return plan
