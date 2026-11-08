"""编排：两条管线共用一套 run 目录约定。

- ``workflow``：perceiver 一次 → 每个 section 循环（writer → reviewer → 不合格就带着上一版重来）。
- ``rag``：perceiver 一次 → 每个 section 先让 writer 出初版，**用初版译文检索一次**，
  然后循环（mimic 改写 → reviewer → 不合格只回 mimic 改写，检索结果复用）。

**一次 ``run_song`` 产生一个 run 目录**：

    data/songs/<song_id>/<workflow|workflow_rag>/run-<ts>-<hash>/
        perceiver/                              # 整曲一次
        sections/001/writer/                    # workflow：这一段最终一轮的译词
        sections/001/mimic_lookup/              # rag：检索参数与命中诗句
        sections/001/mimic_write/               # rag：这一段最终一轮的译词
        sections/001/reviewer/                  # 规则检查（两条管线同名）
        lyrics.zh.txt / lyrics.md

每个 agent 目录固定 ``request.json`` / ``output.json`` / ``meta.json`` 三个文件；
多轮返修只保留**最终一轮**（返修过程仍能从该轮 request 里的反馈和 ``prev_zh`` 看到）。
``--only`` / ``--skip-done`` 会把没重跑的段落整份抄进新 run，所以每个 run 都是完整快照。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from . import checks
from .agents import (
    build_lookup_request,
    build_mimic_request,
    build_perceiver_request,
    build_writer_request,
    parse_draft_output,
    parse_perceiver_output,
)
from .export import write_lyrics
from .llm import LLMClient, LLMError, NeedsLLM, extract_json_parsed
from .models import Analysis, Draft, LookupResult, PoetryHit, Review, SchemaError, Song
from .prompting import prompt_hash
from .song import load_song
from .store import Request, RunStore, run_base, section_index
from .view import AGENTS_BY_PIPELINE, PIPELINES, build_view, load_draft, load_review

__all__ = [
    "PIPELINES",
    "PIPELINE_DIR",
    "AGENTS_BY_PIPELINE",
    "DEFAULT_RAG_K",
    "Step",
    "compose_request",
    "SongOutcome",
    "SectionOutcome",
    "run_song",
    "plan_requests",
    "store_for",
]

DEFAULT_ROOT = Path("data/songs")
DEFAULT_MAX_ROUNDS = 3
#: 检索条数默认 4（与 dubbing.rag.retriever 保持一致；这里不 import rag，免得拖进 chromadb）。
DEFAULT_RAG_K = 4

#: 每条管线的 run 目录名。
PIPELINE_DIR = {"workflow": "workflow", "rag": "workflow_rag"}

REVIEW_AGENT = "reviewer"


class RagError(RuntimeError):
    """rag 管线缺检索器或索引不可用。"""


class PoetryRetriever(Protocol):
    """检索器只要满足这一条协议；测试可以塞一个假的进来。"""

    k: int
    collection: str
    index_id: str

    def search(
        self, query_text: str, *, k: int | None = None, exclude: set[str] | None = None
    ) -> list[PoetryHit]: ...


def store_for(
    root: Path | str | None, song_id: str, pipeline: str = "workflow"
) -> RunStore:
    if pipeline not in PIPELINES:
        raise KeyError(f"未知管线 {pipeline!r}，可选 {PIPELINES}")
    return RunStore(run_base(root or DEFAULT_ROOT, song_id, PIPELINE_DIR[pipeline]))


@dataclass(frozen=True)
class Step:
    """编排器走到哪一步了。``phase`` 取 ``start`` / ``done``。

    ``agent`` 取 ``perceiver`` / ``writer`` / ``mimic`` / ``reviewer``；mimic 的检索与写作
    用 ``note`` 区分。
    """

    agent: str
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
    pipeline: str = "workflow"
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
    llm_agent: str | None = None,
) -> str:
    """写 request.json → 调模型。失败时把原因和已落盘位置写进 meta.json。

    ``agent`` 是落盘目录名（如 ``mimic_write``），``llm_agent`` 是给客户端看的角色名
    （如 ``mimic``，用来选温度）。两者不一致时以 ``llm_agent`` 为准。
    """
    directory = store.prepare(run_id, agent, request, section_index_=section_index_)
    role = llm_agent or agent
    try:
        return llm.complete(agent=role, system=request.system, user=request.user)
    except NeedsLLM as exc:
        store.abandon(run_id, agent, reason="needs-llm", section_index_=section_index_)
        raise NeedsLLM(f"{exc}\n本次入参已落盘：{directory / 'request.json'}") from exc
    except LLMError as exc:
        store.abandon(run_id, agent, reason=f"error: {exc}", section_index_=section_index_)
        raise LLMError(f"{exc}\n本次入参已落盘：{directory / 'request.json'}") from exc


# --------------------------------------------------------------------------
# perceiver
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

    if not force and store.has_output(source_run, "perceiver"):
        store.copy_record(source_run, run_id, "perceiver")  # type: ignore[arg-type]
        analysis = Analysis.from_json(store.read_output(run_id, "perceiver"))
        missing = [sid for sid in song.section_ids if sid not in analysis.sections]
        warnings: list[str] = []
        if missing:
            warnings.append(
                f"perceiver 的产物没有覆盖 {missing}（song.yaml 改过？）——"
                "这些段会缺少情绪参考，建议 --redo-analysis"
            )
            emit(Step("perceiver", "done", note=f"复用上一版，缺 {'/'.join(missing)}"))
        else:
            emit(Step("perceiver", "done", note="复用上一版"))
        return analysis, True, warnings

    emit(Step("perceiver", "start", note="体会整首歌"))
    request = build_perceiver_request(song)
    raw = _invoke(store, run_id, "perceiver", llm, request)
    try:
        parsed = extract_json_parsed(raw)
        analysis, warnings = parse_perceiver_output(parsed.value, song)
    except (LLMError, SchemaError) as exc:
        store.abandon(run_id, "perceiver", reason=f"解析失败：{exc}", raw_response=raw)
        raise
    store.finish(
        run_id,
        "perceiver",
        analysis.to_json(),
        meta=_meta(
            "perceiver", llm, request.system, warnings=warnings, json_repaired=parsed.repaired
        ),
        raw_response=raw,
    )
    emit(Step("perceiver", "done", note=f"{len(analysis.sections)} 段"))
    return analysis, False, warnings


# --------------------------------------------------------------------------
# 每个 section
# --------------------------------------------------------------------------


def _accepted_texts(
    song: Song, store: RunStore, run_id: str, focus: str, pipeline: str
) -> dict[str, str]:
    view = build_view(song, store, run_id, pipeline=pipeline)
    return view.accepted_texts(exclude={p.id for p in song.phrases_of(focus)})


def _review_request(song: Song, section: str, draft: Draft, prev: Draft | None) -> Request:
    data = {
        "section": section,
        "phrases": [p.to_json() for p in song.phrases_of(section)],
        "draft": draft.to_json(),
        "prev_draft": prev.to_json() if prev is not None else None,
    }
    return Request(system=f"({REVIEW_AGENT} 是规则检查，不调用模型)", user="", data=data)


def _call_draft(
    store: RunStore,
    run_id: str,
    agent: str,
    llm: LLMClient,
    request: Request,
    *,
    index: int,
    song: Song,
    section: str,
    llm_agent: str | None = None,
    **meta: object,
) -> tuple[Draft, list[str]]:
    """调一次写词人（writer 或 mimic 写作），解析并落盘。"""
    raw = _invoke(
        store, run_id, agent, llm, request, section_index_=index, llm_agent=llm_agent
    )
    try:
        parsed = extract_json_parsed(raw)
        draft, warnings = parse_draft_output(parsed.value, song, section)
    except (LLMError, SchemaError) as exc:
        store.abandon(
            run_id, agent, reason=f"解析失败：{exc}", raw_response=raw, section_index_=index
        )
        raise
    store.finish(
        run_id,
        agent,
        draft.to_json(),
        meta=_meta(
            llm_agent or agent,
            llm,
            request.system,
            warnings=warnings,
            json_repaired=parsed.repaired,
            **meta,
        ),
        raw_response=raw,
        section_index_=index,
    )
    return draft, warnings


def _outcome(
    section: str, review: Review, rounds: int, warnings: list[str]
) -> SectionOutcome:
    soft = len(review.violations) - len(review.hard)
    return SectionOutcome(
        section=section,
        status="passed" if review.verdict == "pass" else "exhausted",
        rounds=rounds,
        hard=len(review.hard),
        soft=soft,
        regressions=review.regressions,
        warnings=list(warnings),
    )


def _lookup(
    song: Song,
    store: RunStore,
    run_id: str,
    index: int,
    draft: Draft,
    *,
    focus: str,
    retriever: PoetryRetriever | None,
    k: int,
    exclude: set[str] | None = None,
    used: set[str] | None = None,
    on_step: StepHook | None = None,
) -> LookupResult:
    """mimic 的检索部分：不调模型，但要落盘查询、命中与索引血缘。

    基础设施故障（索引缺失、嵌入服务不可用）**显式失败**，不退化成"没有参考"——
    否则会安静地跑出一批"声称用了 RAG 其实没检索"的结果。
    """
    collection = getattr(retriever, "collection", "") if retriever is not None else ""
    request = build_lookup_request(song, draft, focus=focus, k=k, collection=collection)
    store.prepare(run_id, "mimic_lookup", request, section_index_=index)
    query_text = str(request.data.get("query_text") or "")

    def emit(step: Step) -> None:
        if on_step is not None:
            on_step(step)

    if not query_text.strip():
        result = LookupResult(
            section=focus, query_text="", k=k, collection=collection, skipped="empty-query"
        )
        store.finish(
            run_id,
            "mimic_lookup",
            result.to_json(),
            meta={"rule_based": True, "skipped": "empty-query"},
            section_index_=index,
        )
        emit(Step("mimic", "done", section=focus, note="初版为空，跳过检索"))
        return result

    if retriever is None:
        store.abandon(run_id, "mimic_lookup", reason="no-retriever", section_index_=index)
        raise RagError(
            "rag 管线需要检索器：先建库（scripts/build_poetry_index.py），"
            "并设置 DUBBING_RAG_DB / DUBBING_EMBED_URL"
        )
    try:
        hits = retriever.search(query_text, k=k, exclude=exclude)
    except Exception as exc:  # 嵌入/向量库故障：留下证据再抛，不退化成"没有参考"
        store.abandon(run_id, "mimic_lookup", reason=f"error: {exc}", section_index_=index)
        raise RagError(f"检索失败：{exc}") from exc

    result = LookupResult(
        section=focus,
        query_text=query_text,
        k=k,
        hits=tuple(hits),
        collection=collection,
        index_id=str(getattr(retriever, "index_id", "") or ""),
    )
    store.finish(
        run_id,
        "mimic_lookup",
        result.to_json(),
        meta={
            "rule_based": True,
            "collection": result.collection,
            "index_id": result.index_id,
            "k": k,
            "hits": len(result.hits),
        },
        section_index_=index,
    )
    if used is not None:
        used.update(h.chunk_id for h in result.hits)
    emit(Step("mimic", "done", section=focus, note=f"检索到 {len(result.hits)} 条诗句"))
    return result


def attempt_section(
    song: Song,
    store: RunStore,
    run_id: str,
    source_run: str | None,
    analysis: Analysis,
    llm: LLMClient,
    section: str,
    *,
    pipeline: str = "workflow",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    skip_done: bool = False,
    fresh: bool = True,
    retriever: PoetryRetriever | None = None,
    k: int = DEFAULT_RAG_K,
    dedupe_peers: bool = False,
    used_chunks: set[str] | None = None,
    writer_retry: bool = False,
    on_step: StepHook | None = None,
) -> SectionOutcome:
    index = section_index(song, section)

    def emit(step: Step) -> None:
        if on_step is not None:
            on_step(step)

    def run_reviewer(draft: Draft, prev_draft: Draft | None, prev_review: Review | None) -> Review:
        review = checks.review(
            song,
            section,
            draft,
            draft_run=run_id,
            prev_draft=prev_draft,
            prev_review=prev_review,
        )
        store.prepare(
            run_id, REVIEW_AGENT, _review_request(song, section, draft, prev_draft), section_index_=index
        )
        store.finish(
            run_id, REVIEW_AGENT, review.to_json(), meta={"rule_based": True}, section_index_=index
        )
        soft = len(review.violations) - len(review.hard)
        emit(
            Step(
                REVIEW_AGENT,
                "done",
                section=section,
                note=f"{review.verdict}（{len(review.hard)} 硬 / {soft} 软）",
            )
        )
        return review

    if not song.translatable_of(section):
        emit(Step("writer", "done", section=section, note="整段保留原文，跳过"))
        return SectionOutcome(section=section, status="skipped")

    if skip_done and store.has_output(source_run, REVIEW_AGENT, index):
        review = Review.from_json(store.read_output(source_run, REVIEW_AGENT, index))  # type: ignore[arg-type]
        if review.verdict == "pass":
            for agent in AGENTS_BY_PIPELINE[pipeline]:
                store.copy_record(source_run, run_id, agent, section_index_=index)  # type: ignore[arg-type]
            emit(Step("writer", "done", section=section, note="已完成，跳过"))
            return SectionOutcome(section=section, status="reused")

    # fresh：忽略这一段已有的译文，从零开始（旧 run 仍在磁盘上，不会被删）
    prev_draft = None if fresh else load_draft(store, source_run, index, pipeline=pipeline)
    prev_review = None if fresh else load_review(store, source_run, index)

    if pipeline == "workflow":
        result = SectionOutcome(section=section, status="exhausted")
        for attempt in range(1, max_rounds + 1):
            request = build_writer_request(
                song,
                analysis,
                focus=section,
                accepted=_accepted_texts(song, store, run_id, section, pipeline),
                prev_texts=prev_draft.texts() if prev_draft is not None else None,
                feedback=prev_review.violations if prev_review is not None else (),
            )
            emit(Step("writer", "start", section=section, note=f"第 {attempt} 轮"))
            draft, warnings = _call_draft(
                store,
                run_id,
                "writer",
                llm,
                request,
                index=index,
                song=song,
                section=section,
                round=attempt,
            )
            review = run_reviewer(draft, prev_draft, prev_review)
            result = _outcome(section, review, attempt, warnings)
            if review.verdict == "pass":
                return result
            prev_draft, prev_review = draft, review
        return result

    # ---- rag：writer 出初版 → 检索一次 → mimic 改写与返修 ----
    references: LookupResult | None = None
    result = SectionOutcome(section=section, status="exhausted")
    writer_attempts = 2 if writer_retry else 1
    for writer_attempt in range(1, writer_attempts + 1):
        request = build_writer_request(
            song,
            analysis,
            focus=section,
            accepted=_accepted_texts(song, store, run_id, section, pipeline),
            feedback=prev_review.violations if (writer_attempt > 1 and prev_review is not None) else (),
        )
        emit(
            Step(
                "writer",
                "start",
                section=section,
                note="出初版" if writer_attempt == 1 else "带着评审意见重出初版",
            )
        )
        draft0, _warnings = _call_draft(
            store,
            run_id,
            "writer",
            llm,
            request,
            index=index,
            song=song,
            section=section,
            writer_attempt=writer_attempt,
        )
        if references is None:
            # 检索只做一次，返修轮次复用同一批参考
            references = _lookup(
                song,
                store,
                run_id,
                index,
                draft0,
                focus=section,
                retriever=retriever,
                k=k,
                exclude=used_chunks if dedupe_peers else None,
                used=used_chunks,
                on_step=on_step,
            )

        for attempt in range(1, max_rounds + 1):
            request = build_mimic_request(
                song,
                analysis,
                focus=section,
                draft=draft0,
                references=references,
                accepted=_accepted_texts(song, store, run_id, section, pipeline),
                prev_texts=prev_draft.texts() if prev_draft is not None else None,
                feedback=prev_review.violations if prev_review is not None else (),
            )
            emit(Step("mimic", "start", section=section, note=f"改写 第 {attempt} 轮"))
            draft, warnings = _call_draft(
                store,
                run_id,
                "mimic_write",
                llm,
                request,
                index=index,
                song=song,
                section=section,
                llm_agent="mimic",
                round=attempt,
                index_id=references.index_id,
                hits=len(references.hits),
            )
            review = run_reviewer(draft, prev_draft, prev_review)
            result = _outcome(section, review, attempt, warnings)
            if review.verdict == "pass":
                return result
            prev_draft, prev_review = draft, review

    return result


# --------------------------------------------------------------------------
# 整首歌
# --------------------------------------------------------------------------


def run_song(
    song_id: str,
    llm: LLMClient,
    *,
    root: Path | None = None,
    pipeline: str = "workflow",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    skip_done: bool = False,
    fresh: bool = True,
    redo_analysis: bool = False,
    only: set[str] | None = None,
    writer_retry: bool = False,
    rag_k: int | None = None,
    rag_db: Path | str | None = None,
    rag_collection: str | None = None,
    embed_url: str | None = None,
    embed_model: str | None = None,
    embed_local: bool = False,
    dedupe_peers: bool = False,
    retriever: PoetryRetriever | None = None,
    on_step: StepHook | None = None,
) -> SongOutcome:
    if pipeline not in PIPELINES:
        raise KeyError(f"未知管线 {pipeline!r}，可选 {PIPELINES}")
    song = load_song(song_id, root=root)
    if only:
        unknown = sorted(only - set(song.section_ids))
        if unknown:
            raise KeyError(f"没有这些 section：{unknown}，可选 {list(song.section_ids)}")
    store = store_for(root, song_id, pipeline)
    source_run = store.latest_run()

    k = rag_k if rag_k is not None else DEFAULT_RAG_K
    if pipeline == "rag" and retriever is None:
        # 延后 import：workflow 管线不需要 chromadb
        from .rag.embedding import EmbeddingError
        from .rag.retriever import build_retriever
        from .rag.store import StoreError

        try:
            retriever = build_retriever(
                db=rag_db,
                collection=rag_collection,
                k=k,
                embed_url=embed_url,
                embed_model=embed_model,
                local=embed_local,
            )
        except (StoreError, EmbeddingError) as exc:
            raise RagError(str(exc)) from exc
    if retriever is not None and rag_k is None:
        k = int(getattr(retriever, "k", DEFAULT_RAG_K))

    run_id = store.new_run()
    analysis, reused, warnings = ensure_analysis(
        song, store, run_id, source_run, llm, force=redo_analysis, on_step=on_step
    )
    outcome = SongOutcome(song_id=song_id, pipeline=pipeline, run_id=run_id, analysis_reused=reused)
    outcome.warnings.extend(warnings)

    used_chunks: set[str] = set()
    for section in song.sections:
        index = section_index(song, section.id)
        if only is not None and section.id not in only:
            # 没被点名的段落整份抄过来，保证这个 run 仍是完整快照
            if source_run is not None:
                for agent in AGENTS_BY_PIPELINE[pipeline]:
                    store.copy_record(source_run, run_id, agent, section_index_=index)
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
                pipeline=pipeline,
                max_rounds=max_rounds,
                skip_done=skip_done,
                fresh=fresh,
                retriever=retriever,
                k=k,
                dedupe_peers=dedupe_peers,
                used_chunks=used_chunks,
                writer_retry=writer_retry,
                on_step=on_step,
            )
        )

    view = build_view(song, store, run_id, pipeline=pipeline)
    outcome.lyrics_paths = write_lyrics(song, view, store.run_dir(run_id))
    store.set_latest(
        run_id,
        song=song_id,
        pipeline=pipeline,
        status="completed" if outcome.ok else "incomplete",
        model=getattr(llm, "model", llm.name),
    )
    return outcome


def compose_request(
    song_id: str,
    *,
    agent: str = "writer",
    section: str | None = None,
    root: Path | None = None,
    pipeline: str = "workflow",
    fresh: bool = True,
) -> Request:
    """现拼一份“如果现在要调用，会发出去什么”的请求（不调用模型）。

    跟真实调用唯一的差别：返修轮次里 reviewer 的结论不会出现，因为这里没有“上一轮”；
    rag 管线的 mimic 也只能给出空的初版与空参考（它们要等 writer 与检索跑完）。
    想看带反馈的原样请求，用 ``--run <id>`` 读 ``sections/NNN/.../request.json``。
    """
    song = load_song(song_id, root=root)
    store = store_for(root, song_id, pipeline)
    source_run = store.latest_run()

    if agent == "perceiver":
        return build_perceiver_request(song)

    if section is None:
        section = next((s.id for s in song.sections if song.translatable_of(s.id)), None)
        if section is None:
            raise KeyError("这首歌没有可译的 section")
    if section not in song.section_ids:
        raise KeyError(f"未知 section {section!r}，可选 {list(song.section_ids)}")

    analysis = (
        Analysis.from_json(store.read_output(source_run, "perceiver"))  # type: ignore[arg-type]
        if store.has_output(source_run, "perceiver")
        else Analysis(theme="", emotional_arc="", narrative="")
    )
    index = section_index(song, section)
    prev = None if fresh else load_draft(store, source_run, index, pipeline=pipeline)
    accepted = _accepted_texts(song, store, source_run, section, pipeline) if source_run else {}
    prev_texts = prev.texts() if prev is not None else None

    if agent == "mimic":
        return build_mimic_request(
            song,
            analysis,
            focus=section,
            draft=Draft(section=section, phrases=()),
            references=None,
            accepted=accepted,
            prev_texts=prev_texts,
        )
    return build_writer_request(
        song, analysis, focus=section, accepted=accepted, prev_texts=prev_texts
    )


def plan_requests(
    song_id: str,
    *,
    root: Path | None = None,
    pipeline: str = "workflow",
    skip_done: bool = False,
    fresh: bool = True,
) -> list[tuple[str, Request]]:
    """渲染出所有待发请求但不调用模型（给 ``--dry-run`` 用）。

    rag 管线的 ``mimic_lookup`` 依赖真实向量库、``mimic_write`` 依赖 writer 的初版与检索结果，
    这里只能渲染出空壳，仅供人工核对提示词与字段形状。
    """
    song = load_song(song_id, root=root)
    store = store_for(root, song_id, pipeline)
    source_run = store.latest_run()

    plan: list[tuple[str, Request]] = []
    if not store.has_output(source_run, "perceiver"):
        plan.append(("perceiver", build_perceiver_request(song)))
        # 没有 perceiver 的产物时，写词人的 analysis 只能留空
        analysis = Analysis(theme="", emotional_arc="", narrative="")
    else:
        analysis = Analysis.from_json(store.read_output(source_run, "perceiver"))  # type: ignore[arg-type]

    for section in song.sections:
        if not song.translatable_of(section.id):
            continue
        index = section_index(song, section.id)
        if skip_done and store.has_output(source_run, REVIEW_AGENT, index):
            review = Review.from_json(store.read_output(source_run, REVIEW_AGENT, index))  # type: ignore[arg-type]
            if review.verdict == "pass":
                continue
        prev = None if fresh else load_draft(store, source_run, index, pipeline=pipeline)
        accepted = _accepted_texts(song, store, source_run, section.id, pipeline) if source_run else {}
        plan.append(
            (
                f"writer-{section.id}",
                build_writer_request(
                    song,
                    analysis,
                    focus=section.id,
                    accepted=accepted,
                    prev_texts=prev.texts() if prev is not None else None,
                ),
            )
        )
        if pipeline == "rag":
            plan.append(
                (
                    f"mimic-{section.id}",
                    build_mimic_request(
                        song,
                        analysis,
                        focus=section.id,
                        draft=Draft(section=section.id, phrases=()),
                        references=None,
                        accepted=accepted,
                    ),
                )
            )
    return plan
