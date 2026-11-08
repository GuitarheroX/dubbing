"""命令行入口。

    python -m dubbing songs
    python -m dubbing run   --song kotozute
    python -m dubbing run   --song kotozute --pipeline rag --client api
    python -m dubbing run   --song kotozute --dry-run
    python -m dubbing view  --song kotozute

一次 ``run`` 在 ``data/songs/<id>/<pipeline>/`` 下产生一个 run 目录；旧 run 不会被
覆盖，``latest.json`` 决定 ``view`` / ``export`` 默认看哪一次。

``--pipeline`` 取 ``workflow``（perceiver → writer → reviewer）或 ``rag``
（perceiver → writer → mimic 检索/改写 → reviewer）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .counting import count_syllables
from .export import write_lyrics
from .llm import DEFAULT_MAX_TOKENS, LLMError, NeedsLLM, build_client, load_dotenv
from .models import SchemaError, Violation
from .orchestrator import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_RAG_K,
    DEFAULT_ROOT,
    RagError,
    Step,
    compose_request,
    plan_requests,
    run_song,
    store_for,
)
from .report import AGENT_LABELS, list_runs, render_record, table
from .song import list_songs, load_song
from .store import section_index
from .view import AGENTS_BY_PIPELINE, PIPELINES, build_view

__all__ = ["main"]

PIPELINE_HELP = "workflow=perceiver→writer→reviewer；rag=多一个 mimic（检索现代诗 + 改写）"


# -- 子命令 --------------------------------------------------------------


def cmd_songs(args: argparse.Namespace) -> int:
    ids = list_songs(root=Path(args.root))
    if not ids:
        print(f"{args.root} 下没有歌")
        return 1
    root = Path(args.root)
    for sid in ids:
        song = load_song(sid, root=root)
        n = len(song.phrases)
        total = sum(p.budget.max for p in song.phrases if p.budget is not None)
        wf = len(store_for(root, sid, "workflow").list_runs())
        rag = len(store_for(root, sid, "rag").list_runs())
        print(
            f"{sid:<12} {song.title:<10} {len(song.sections)} 段 / {n} 句 / 预算共 {total} 字"
            f" / workflow {wf} run / rag {rag} run"
        )
    return 0


def _step_printer() -> "callable":
    """把编排器的每一步打到 stderr，stdout 留给最终结果。"""

    def emit(step: Step) -> None:
        label = AGENT_LABELS.get(step.agent, step.agent)
        where = f" {step.section}" if step.section else ""
        if step.phase == "start":
            print(f"[{label}]{where} {step.note or '进行中'} …", file=sys.stderr, flush=True)
        else:
            print(f"[{label}]{where} {step.note or '完成'}", file=sys.stderr, flush=True)

    return emit


def _resolve_run(store, args, song) -> str | None:
    """``--run`` 优先，否则用 ``latest.json`` 指的那一次。"""
    if getattr(args, "run", None):
        return args.run
    return store.latest_run()


def cmd_view(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = store_for(Path(args.root), args.song, args.pipeline)
    run_id = _resolve_run(store, args, song)
    view = build_view(song, store, run_id, pipeline=args.pipeline)

    rows: list[list[str]] = [["id", "ja", "预算", "译词", "字", "状态"]]
    for r in view.rows:
        if r.budget is None:
            budget = "-"
        elif r.budget.min == r.budget.max:
            budget = str(r.budget.min)
        else:
            budget = f"{r.budget.min}-{r.budget.max}"
        rows.append(
            [
                r.phrase_id,
                r.ja,
                budget,
                r.sing_text or "",
                "" if r.lang is None or r.verdict == "skip" else str(count_syllables(r.sing_text)),
                r.verdict,
            ]
        )
    print(table(rows))

    done = sum(1 for r in view.rows if r.verdict in {"pass", "skip"})
    print(f"\n{done}/{len(view.rows)} 句已定稿（pass 或 skip）")
    print(f"管线：{args.pipeline}　run：{run_id or '（还没有 run）'}")
    print(
        f"看成品歌词：python -m dubbing export --song {args.song} --pipeline {args.pipeline}"
    )
    if args.json and run_id:
        out = store.run_dir(run_id) / "view.json"
        out.write_text(
            json.dumps(view.to_json(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"已写出 {out}")
    return 0


def _print_violations(violations: list[Violation]) -> None:
    for v in violations:
        exp = "" if v.expected is None else f" 期望 {v.expected[0]}-{v.expected[1]}"
        act = "" if v.actual is None else f" 实际 {v.actual}"
        fix = f" | {v.fix}" if v.fix else ""
        print(f"    [{v.severity}] {v.phrase_id} {v.code}{exp}{act}{fix}")


def cmd_runs(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = store_for(Path(args.root), args.song, args.pipeline)
    rows = list_runs(store, song)
    if not rows:
        print(
            f"{args.pipeline} 管线还没有任何 run。先跑一次："
            f"python -m dubbing run --song {args.song} --pipeline {args.pipeline} --client api"
        )
        return 0

    out = [["", "run_id", "模型", "perceiver", "段落", "歌词", "tokens", "状态"]]
    for r in rows:
        out.append(
            [
                "*" if r["current"] else "",
                r["run_id"],
                r["model"],
                "有" if r["has_perceiver"] else "—",
                f"{r['passed']}/{r['total']} pass",
                "有" if r["lyrics"] else "—",
                "" if r["out_tokens"] is None else str(r["out_tokens"]),
                r["status"],
            ]
        )
    print(f"{song.title}（{args.song}，{args.pipeline}）")
    print(table(out))
    print("\n* = latest.json 当前指向的一次")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = store_for(Path(args.root), args.song, args.pipeline)
    run_id = _resolve_run(store, args, song)

    if not run_id:
        print(
            f"还没有任何 run。先跑一次："
            f"python -m dubbing run --song {args.song} --pipeline {args.pipeline} --client api"
        )
        return 0

    sections = [args.section] if args.section else None
    if sections:
        missing = [s for s in sections if s not in song.section_ids]
        if missing:
            print(f"没有这些 section：{missing}", file=sys.stderr)
            return 1

    agents = [args.agent] if args.agent else ["perceiver", *AGENTS_BY_PIPELINE[args.pipeline]]
    targets = sections or list(song.section_ids)
    printed = False
    for agent in agents:
        if agent == "perceiver":
            print(render_record(song, store, run_id, agent, None), end="")
            printed = True
            continue
        for sid in targets:
            print(render_record(song, store, run_id, agent, sid), end="")
            printed = True

    if not printed:
        print("还没有产物。先跑一次：python -m dubbing run --song <id> --client api")
    return 0


#: ``prompt`` 的 agent 名 → 磁盘上的产物目录（mimic 的写作产物在 mimic_write）。
PROMPT_AGENT_DIR = {"perceiver": "perceiver", "writer": "writer", "mimic": "mimic_write"}


def cmd_prompt(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = store_for(Path(args.root), args.song, args.pipeline)
    agent = args.agent or "writer"

    if args.run:
        directory = PROMPT_AGENT_DIR.get(agent, agent)
        section = args.section
        if directory != "perceiver" and section is None:
            section = next((s for s in song.section_ids if song.translatable_of(s)), None)
        index = section_index(song, section) if (directory != "perceiver" and section) else None
        payload = store.read_request(args.run, directory, index)
        meta = store.read_meta(args.run, directory, index)
        system = payload["messages"]["system"]
        user = payload["messages"]["user"]
        where = f"，section {section}" if section else ""
        head = f"# {args.run}（{args.pipeline} / {directory}{where}）真实发出去的请求"
        if meta.get("model"):
            head += f"，model {meta['model']}"
        if agent == "mimic":
            head += "\n# 注：mimic 的检索入参见 show --agent mimic_lookup"
    else:
        request = compose_request(
            args.song,
            agent=agent,
            section=args.section,
            root=Path(args.root),
            pipeline=args.pipeline,
        )
        system, user = request.system, request.user
        where = f"，section {args.section}" if args.section else "，section 取第一个可译段"
        head = f"# {song.id} 现拼的请求（{args.pipeline} / {agent}{where}）"
        if agent == "mimic":
            head += "\n# 注：现拼的 mimic 请求里初版与参考都是空的，真实请求要跑完 writer 与检索"

    print(head)
    print(f"# system {len(system)} 字 / user {len(user)} 字")
    print()
    print("=" * 20, "system", "=" * 20)
    print(system)
    print()
    print("=" * 21, "user", "=" * 21)
    print(user)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = store_for(Path(args.root), args.song, args.pipeline)
    run_id = _resolve_run(store, args, song)
    if not run_id:
        print("还没有任何 run，没有可导出的歌词。", file=sys.stderr)
        return 1
    view = build_view(song, store, run_id, pipeline=args.pipeline)

    out_dir = Path(args.out) if args.out else store.run_dir(run_id)
    paths = write_lyrics(song, view, out_dir)
    for p in paths:
        print(p)
    print()
    print(paths[0].read_text(encoding="utf-8"))
    return 0


def _run_one(song_id: str, args: argparse.Namespace, root: Path) -> int:
    pipeline = args.pipeline

    if args.dry_run:
        plan = plan_requests(
            song_id, root=root, pipeline=pipeline, skip_done=args.skip_done, fresh=args.fresh
        )
        song = load_song(song_id, root=root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out_dir = store_for(root, song_id, pipeline).base / f"preview-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not plan:
            print("没有待发的请求：perceiver 已产出，各段也都已定稿。")
            return 0
        for name, request in plan:
            stem = name.replace(":", "-")
            (out_dir / f"{stem}.system.md").write_text(request.system + "\n", encoding="utf-8")
            (out_dir / f"{stem}.data.json").write_text(
                json.dumps(request.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"{name:<12} -> {out_dir}/{stem}.system.md  +  {stem}.data.json")
        print("\n--dry-run：没有调用任何模型。")
        if any(name == "perceiver" for name, _ in plan):
            print("注意：perceiver 还没跑，写词人的 analysis 会是空的；先跑通 perceiver 才有意义。")
        if pipeline == "rag":
            print("注意：mimic 的检索要等 writer 的初版，dry-run 渲染不出来；mimic 的入参里初版与参考都是空的。")
        print(f"共 {len(song.phrases)} 句，{len(plan)} 个请求。")
        return 0

    llm = build_client(
        args.client,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        temperature_by_agent={
            k: v
            for k, v in (
                ("perceiver", args.temperature_perceiver),
                ("writer", args.temperature_writer),
                ("mimic", args.temperature_mimic),
            )
            if v is not None
        }
        or None,
        max_tokens=args.max_tokens,
    )
    try:
        outcome = run_song(
            song_id,
            llm,
            root=root,
            pipeline=pipeline,
            max_rounds=args.max_rounds,
            skip_done=args.skip_done,
            fresh=args.fresh,
            redo_analysis=args.redo_analysis,
            only=set(args.only) if args.only else None,
            writer_retry=args.writer_retry,
            rag_k=args.rag_k,
            rag_db=args.rag_db,
            rag_collection=args.rag_collection,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            embed_local=args.embed_local,
            dedupe_peers=args.rag_dedupe_peers,
            on_step=_step_printer(),
        )
    except NeedsLLM as exc:
        print(f"\n停在模型调用处：{exc}\n", file=sys.stderr)
        print("想真的调用模型：", file=sys.stderr)
        print("  export DEEPSEEK_API_KEY=sk-...", file=sys.stderr)
        print("  python -m dubbing run --song <id> --client api", file=sys.stderr)
        return 2
    except RagError as exc:
        print(f"\nrag 管线跑不下去：{exc}\n", file=sys.stderr)
        print("本地索引与嵌入服务的默认值：", file=sys.stderr)
        print("  索引 data/poetry/chroma（--rag-db 可改，先跑 scripts/build_poetry_index.py）", file=sys.stderr)
        print("  嵌入 DUBBING_EMBED_URL（默认 http://127.0.0.1:8080/v1/embeddings）", file=sys.stderr)
        return 1
    except LLMError as exc:
        print(f"\n模型调用出错：{exc}\n", file=sys.stderr)
        print(
            f"查一遍 DEEPSEEK_API_KEY 和 --model；入参在 {pipeline} 的 run 目录里。",
            file=sys.stderr,
        )
        return 1
    except SchemaError as exc:
        print(f"数据格式有问题：{exc}", file=sys.stderr)
        return 1

    print(f"\n{song_id}　{pipeline}　run {outcome.run_id}")
    print(f"  perceiver: {'复用上一版' if outcome.analysis_reused else '新跑'}")
    rows = [["section", "status", "轮次", "hard", "soft"]]
    for s in outcome.sections:
        rows.append(
            [
                s.section,
                s.status,
                "" if s.rounds == 0 else str(s.rounds),
                "" if s.rounds == 0 else str(s.hard),
                "" if s.rounds == 0 else str(s.soft),
            ]
        )
    print()
    print(table(rows))

    for s in outcome.sections:
        for w in s.warnings:
            print(f"  ! {s.section}: {w}")
        if s.regressions:
            print(f"  ! {s.section}: 返修改坏了 {list(s.regressions)}")
    for w in outcome.warnings:
        print(f"  ! {w}")

    print()
    for p in outcome.lyrics_paths:
        print(f"歌词已写出：{p}")

    return 0 if outcome.ok else 1


def cmd_poetry_search(args: argparse.Namespace) -> int:
    """直接查本地索引：调 k、试查询写法、抽查召回质量都靠它。"""
    from .rag.embedding import EmbeddingError
    from .rag.retriever import build_retriever
    from .rag.store import StoreError

    try:
        retriever = build_retriever(
            db=args.rag_db,
            collection=args.rag_collection,
            k=args.k,
            embed_url=args.embed_url,
            embed_model=args.embed_model,
            local=args.embed_local,
        )
        hits = retriever.search(args.query, k=args.k)
    except (StoreError, EmbeddingError) as exc:
        print(f"查不了：{exc}", file=sys.stderr)
        print(
            "先建库：python scripts/build_poetry_index.py --poetry-root data/poetry/raw",
            file=sys.stderr,
        )
        return 1

    print(f"索引 {retriever.index_id}　collection {retriever.collection}　k={args.k}")
    print()
    if not hits:
        print("没有命中。")
        return 0
    for i, hit in enumerate(hits, 1):
        who = f"　{hit.poet}" if hit.poet else ""
        title = f"《{hit.title}》" if hit.title else "（无题）"
        print(f"{i}. [{hit.score:.4f}] {title}{who}　{hit.chunk_id}")
        for line in hit.text.splitlines():
            print(f"     {line}")
        print()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root)

    if args.all_songs:
        targets = list_songs(root=root)
    elif args.song:
        targets = list(args.song)
    else:
        print("要么给 --song <id>（可重复），要么加 --all。", file=sys.stderr)
        ids = list_songs(root=root)
        print("歌库：" + ("、".join(ids) if ids else "（空）"), file=sys.stderr)
        return 2

    if not targets:
        print(f"{root} 下没有歌", file=sys.stderr)
        return 1

    if len(targets) == 1:
        return _run_one(targets[0], args, root)

    failures: list[tuple[str, int]] = []
    for i, song_id in enumerate(targets, 1):
        print(f"\n=== [{i}/{len(targets)}] {song_id} ===", file=sys.stderr)
        code = _run_one(song_id, args, root)
        if code:
            failures.append((song_id, code))
            if code == 2:
                print("后面的歌跳过（同一原因，再跑也是这个结果）", file=sys.stderr)
                break

    print()
    print(f"批量结束：{len(targets) - len(failures)} 首成功 / {len(failures)} 首失败")
    for song_id, code in failures:
        print(f"  ✗ {song_id}（exit {code}）")
    return 1 if failures else 0


def _add_pipeline(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pipeline",
        choices=PIPELINES,
        default=os.environ.get("DUBBING_PIPELINE") or "workflow",
        help=PIPELINE_HELP,
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="dubbing", description="日语歌歌词汉译配流水线")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="歌库目录")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_songs = sub.add_parser("songs", help="列出歌库")
    p_songs.set_defaults(func=cmd_songs)

    p_view = sub.add_parser("view", help="打印当前全曲视图")
    p_view.add_argument("--song", required=True)
    p_view.add_argument("--run", default=None, help="指定 run，默认看 latest.json")
    p_view.add_argument("--json", action="store_true", help="同时把 view.json 写进该 run 目录")
    _add_pipeline(p_view)
    p_view.set_defaults(func=cmd_view)

    p_runs = sub.add_parser("runs", help="列出所有 run（标出 latest.json 指的是哪一次）")
    p_runs.add_argument("--song", required=True)
    _add_pipeline(p_runs)
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="看某个 agent 的产物")
    p_show.add_argument("--song", required=True)
    p_show.add_argument(
        "--agent",
        choices=["perceiver", "writer", "mimic_lookup", "mimic_write", "reviewer"],
        default=None,
        help="不给就按管线的顺序全部打印",
    )
    p_show.add_argument("--section", default=None)
    p_show.add_argument("--run", default=None, help="指定 run，默认看 latest.json")
    _add_pipeline(p_show)
    p_show.set_defaults(func=cmd_show)

    p_prompt = sub.add_parser("prompt", help="看某个 agent 收到的提示词与数据")
    p_prompt.add_argument("--song", required=True)
    p_prompt.add_argument("--agent", choices=["perceiver", "writer", "mimic"], default="writer")
    p_prompt.add_argument("--section", default=None, help="不给就取第一个可译段")
    p_prompt.add_argument("--run", default=None, help="读某次真实调用的 request.json")
    _add_pipeline(p_prompt)
    p_prompt.set_defaults(func=cmd_prompt)

    p_export = sub.add_parser("export", help="导出歌词文件（lyrics.zh.txt / lyrics.md）")
    p_export.add_argument("--song", required=True)
    p_export.add_argument("--run", default=None, help="指定 run，默认看 latest.json")
    p_export.add_argument("--out", default=None, help="输出目录，默认该 run 目录")
    _add_pipeline(p_export)
    p_export.set_defaults(func=cmd_export)

    p_search = sub.add_parser(
        "poetry-search", help="直接查本地现代诗索引（调 k、试查询写法、抽查召回）"
    )
    p_search.add_argument("query", help="查询文本，通常是一段中文译文")
    p_search.add_argument("--k", type=int, default=DEFAULT_RAG_K)
    p_search.add_argument("--rag-db", default=None, help="本地 Chroma 目录，默认 data/poetry/chroma")
    p_search.add_argument("--rag-collection", default=None)
    p_search.add_argument("--embed-url", default=None)
    p_search.add_argument("--embed-model", default=None)
    p_search.add_argument("--embed-local", action="store_true")
    p_search.set_defaults(func=cmd_poetry_search)

    p_run = sub.add_parser(
        "run",
        help="跑一遍，在 <pipeline>/ 下新建一个 run（perceiver/writer/mimic/reviewer 全部从头生成）",
    )
    p_run.add_argument(
        "--song", action="append", default=None, help="歌 id；可重复，也可用 --all"
    )
    p_run.add_argument(
        "--all", action="store_true", dest="all_songs", help="跑歌库里所有歌"
    )
    _add_pipeline(p_run)
    p_run.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p_run.add_argument(
        "--continue",
        dest="fresh",
        action="store_false",
        help="以上一个 run 的译文为起点重跑（默认：从零生成）",
    )
    p_run.add_argument(
        "--skip-done", action="store_true", help="把上一 run 已定稿的段整份抄过来，只补没定稿的段"
    )
    p_run.add_argument(
        "--only", action="append", metavar="SECTION", help="只重跑指定段，其余段从上一 run 抄来"
    )
    p_run.add_argument("--redo-analysis", action="store_true", help="重跑 perceiver")
    p_run.add_argument(
        "--writer-retry",
        action="store_true",
        help="rag：mimic 全部轮次都没过时，再让 writer 带着评审意见重出一次初版（默认关）",
    )
    p_run.add_argument("--dry-run", action="store_true", help="只渲染请求，不调用模型")
    p_run.add_argument(
        "--client",
        choices=["pending", "api"],
        default=os.environ.get("DUBBING_CLIENT") or "pending",
        help="pending=不发请求（默认）；api=走 DeepSeek 的 Anthropic 兼容端点",
    )
    p_run.add_argument("--model", default=None, help="默认取 DUBBING_MODEL 或 deepseek-flash")
    p_run.add_argument(
        "--base-url",
        default=None,
        help="接口根地址，默认取 DUBBING_BASE_URL 或 https://api.deepseek.com/anthropic",
    )
    p_run.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="通用温度，默认 0.7（或 DUBBING_TEMPERATURE）",
    )
    p_run.add_argument(
        "--temperature-perceiver", type=float, default=None, help="只覆盖 perceiver（体会意境）的温度"
    )
    p_run.add_argument(
        "--temperature-writer", type=float, default=None, help="只覆盖 writer（写初版）的温度"
    )
    p_run.add_argument(
        "--temperature-mimic",
        type=float,
        default=None,
        help="只覆盖 mimic（严格改写）的温度，默认 1.5",
    )
    p_run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)

    rag = p_run.add_argument_group("rag 管线")
    rag.add_argument("--rag-k", type=int, default=None, help=f"检索条数，默认 {DEFAULT_RAG_K}")
    rag.add_argument("--rag-db", default=None, help="本地 Chroma 目录，默认 data/poetry/chroma")
    rag.add_argument("--rag-collection", default=None, help="collection 名，默认 poetry_stanza256_v1")
    rag.add_argument(
        "--embed-url",
        default=None,
        help="OpenAI 兼容的嵌入端点，默认 DUBBING_EMBED_URL 或 http://127.0.0.1:8080/v1/embeddings",
    )
    rag.add_argument("--embed-model", default=None, help="嵌入模型名，默认 BAAI/bge-large-zh-v1.5")
    rag.add_argument(
        "--embed-local", action="store_true", help="本机推理（需要 sentence-transformers）"
    )
    rag.add_argument(
        "--rag-dedupe-peers",
        action="store_true",
        help="跨段去重：同一首诗全曲只用一次（默认关，开了会改变召回）",
    )
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if getattr(args, "rag_k", None) is not None and args.rag_k < 1:
        parser.error("--rag-k 必须为正整数")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
