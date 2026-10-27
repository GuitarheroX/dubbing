"""命令行入口。

    python -m dubbing songs
    python -m dubbing run   --song kotozute
    python -m dubbing run   --song kotozute --dry-run
    python -m dubbing view  --song kotozute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .counting import count_syllables
from .export import write_lyrics
from .report import list_runs, render_run, table
from .llm import LLMError, NeedsLLM, build_client, load_dotenv
from .models import SchemaError, Violation
from .orchestrator import (
    DEFAULT_MAX_ROUNDS,
    DEFAULT_ROOT,
    Step,
    plan_requests,
    run_song,
)
from .song import list_songs, load_song
from .store import SongStore
from .view import build_view

__all__ = ["main"]


# -- 终端表格（中日文按两格宽对齐） ---------------------------------------


# -- 子命令 --------------------------------------------------------------


def cmd_songs(args: argparse.Namespace) -> int:
    ids = list_songs(root=Path(args.root))
    if not ids:
        print(f"{args.root} 下没有歌")
        return 1
    for sid in ids:
        song = load_song(sid, root=Path(args.root))
        n = len(song.phrases)
        total = sum(p.budget.max for p in song.phrases if p.budget is not None)
        print(f"{sid:<12} {song.title:<10} {len(song.sections)} 段 / {n} 句 / 预算共 {total} 字")
    return 0


def _step_printer() -> "callable":
    """把编排器的每一步打到 stderr，stdout 留给最终结果。"""

    def emit(step: Step) -> None:
        label = {"a": "A", "b": "B", "c": "C"}.get(step.agent, step.agent.upper())
        where = f" {step.section}" if step.section else ""
        if step.phase == "start":
            print(f"[{label}]{where} {step.note or '进行中'} …", file=sys.stderr, flush=True)
        else:
            print(f"[{label}]{where} {step.note or '完成'}", file=sys.stderr, flush=True)

    return emit


def cmd_view(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = SongStore(Path(args.root), args.song)
    view = build_view(song, store.load_state(), store)

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
    print(f"看成品歌词：python -m dubbing export --song {args.song}")
    if args.json:
        out = Path(args.root) / args.song / "view.json"
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
    store = SongStore(Path(args.root), args.song)
    state = store.load_state()
    rows = list_runs(store, state)
    if not rows:
        print("还没有任何 run。先跑一次：python -m dubbing run --song <id> --client api")
        return 0

    table = [["", "run_id", "agent", "模型", "轮", "tokens", "说明"]]
    for r in rows:
        table.append(
            [
                "*" if r["current"] else "",
                r["run_id"],
                r["agent"].upper(),
                r["model"],
                "" if r["round"] is None else str(r["round"]),
                "" if r["out_tokens"] is None else str(r["out_tokens"]),
                r["note"],
            ]
        )
    print(f"{song.title}（{args.song}）")
    print(table(table))
    print("\n* = 当前视图正在用的一版")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = SongStore(Path(args.root), args.song)
    state = store.load_state()

    if args.run:
        print(render_run(store, song, args.run))
        return 0

    sections = [args.section] if args.section else list(song.section_ids)
    missing = [s for s in sections if s not in song.section_ids]
    if missing:
        print(f"没有这些 section：{missing}", file=sys.stderr)
        return 1

    printed = False
    want_a = args.agent in (None, "a")
    if want_a and store.has_run(state.analysis):
        print(render_run(store, song, state.analysis))
        printed = True

    if args.agent != "a":
        for sid in sections:
            ss = state.sections.get(sid)
            if ss is None:
                continue
            want = args.agent
            if want in (None, "b") and store.has_run(ss.draft):
                print(render_run(store, song, ss.draft))
                printed = True
            if want in (None, "c") and store.has_run(ss.review):
                print(render_run(store, song, ss.review))
                printed = True

    if not printed:
        print("还没有产物。先跑一次：python -m dubbing run --song <id> --client api")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = SongStore(Path(args.root), args.song)
    view = build_view(song, store.load_state(), store)

    out_dir = Path(args.out) if args.out else Path(args.root) / args.song
    paths = write_lyrics(song, view, out_dir)
    for p in paths:
        print(p)
    print()
    print(paths[0].read_text(encoding="utf-8"))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.root)

    if args.dry_run:
        plan = plan_requests(
            args.song, root=root, skip_done=args.skip_done, fresh=args.fresh
        )
        song = load_song(args.song, root=root)
        out_dir = root / args.song / "preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not plan:
            print("没有待发的请求：A 已产出，各段也都已定稿。")
            return 0
        for name, request in plan:
            stem = name.replace(":", "-")
            (out_dir / f"{stem}.system.md").write_text(request.system + "\n", encoding="utf-8")
            (out_dir / f"{stem}.data.json").write_text(
                json.dumps(request.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"{name:<8} -> {out_dir}/{stem}.system.md  +  {stem}.data.json")
        print("\n--dry-run：没有调用任何模型。")
        if any(name == "a" for name, _ in plan):
            print("注意：A 还没跑，B 的 analysis 会是空的；先跑通 A 才有意义。")
        print(f"共 {len(song.phrases)} 句，{len(plan)} 个请求。")
        return 0

    store = SongStore(root, args.song)
    pending = args.client == "pending"
    full_fresh = not args.only and args.fresh and not args.skip_done
    if full_fresh and not pending:
        info = store.reset(keep_history=args.keep_history)
        if info["archive"]:
            print(
                f"旧结果已移入 {info['archive']}（{info['runs']} 个 run）",
                file=sys.stderr,
            )
        else:
            print(
                f"已删除旧结果：{info['runs']} 个 run"
                + ("、state.yaml" if info["state"] else ""),
                file=sys.stderr,
            )
    elif full_fresh and pending:
        print(
            "pending 客户端不清空旧结果（会白删）。真要重来请加 --client api。",
            file=sys.stderr,
        )

    llm = build_client(
        args.client,
        model=args.model,
        temperature=args.temperature,
        temperature_by_agent={
            k: v
            for k, v in (("a", args.temperature_a), ("b", args.temperature_b))
            if v is not None
        }
        or None,
        max_tokens=args.max_tokens,
    )
    try:
        outcome = run_song(
            args.song,
            llm,
            root=root,
            max_rounds=args.max_rounds,
            skip_done=args.skip_done,
            fresh=args.fresh,
            redo_analysis=args.redo_analysis,
            only=set(args.only) if args.only else None,
            on_step=_step_printer(),
        )
    except NeedsLLM as exc:
        print(f"\n停在模型调用处：{exc}\n", file=sys.stderr)
        print("想真的调用模型：", file=sys.stderr)
        print("  export DEEPSEEK_API_KEY=sk-...", file=sys.stderr)
        print("  python -m dubbing run --song <id> --client api", file=sys.stderr)
        return 2
    except LLMError as exc:
        print(f"\n模型调用出错：{exc}\n", file=sys.stderr)
        print("查一遍 DEEPSEEK_API_KEY 和 --model；入参在 runs/<id>/request.json。", file=sys.stderr)
        return 1
    except SchemaError as exc:
        print(f"数据格式有问题：{exc}", file=sys.stderr)
        return 1

    print(f"\n{args.song}")
    print(f"  A: {outcome.analysis_run}（{'复用' if outcome.analysis_reused else '新跑'}）")
    rows = [["section", "status", "轮次", "hard", "soft", "draft"]]
    for s in outcome.sections:
        rows.append(
            [
                s.section,
                s.status,
                "" if s.rounds == 0 else str(s.rounds),
                "" if s.rounds == 0 else str(s.hard),
                "" if s.rounds == 0 else str(s.soft),
                s.draft_run or "",
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

    song = load_song(args.song, root=root)
    store = SongStore(root, args.song)
    view = build_view(song, store.load_state(), store)
    paths = write_lyrics(song, view, root / args.song)
    print()
    for p in paths:
        print(f"歌词已写出：{p}")

    return 0 if outcome.ok else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="dubbing", description="日语歌歌词汉译配流水线")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="歌库目录")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_songs = sub.add_parser("songs", help="列出歌库")
    p_songs.set_defaults(func=cmd_songs)

    p_view = sub.add_parser("view", help="打印当前全曲视图")
    p_view.add_argument("--song", required=True)
    p_view.add_argument("--json", action="store_true", help="同时写出 view.json")
    p_view.set_defaults(func=cmd_view)

    p_runs = sub.add_parser("runs", help="列出所有 run（标出当前用的是哪一版）")
    p_runs.add_argument("--song", required=True)
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="看某个 agent 的产物")
    p_show.add_argument("--song", required=True)
    p_show.add_argument("--agent", choices=["a", "b", "c"], default=None)
    p_show.add_argument("--section", default=None)
    p_show.add_argument("--run", default=None, help="直接看某次 run 的产物")
    p_show.set_defaults(func=cmd_show)

    p_export = sub.add_parser("export", help="导出歌词文件（lyrics.zh.txt / lyrics.md）")
    p_export.add_argument("--song", required=True)
    p_export.add_argument("--out", default=None, help="输出目录，默认 data/songs/<id>/")
    p_export.set_defaults(func=cmd_export)

    p_run = sub.add_parser(
        "run",
        help="跑一遍（默认清空旧结果、A/B/C 全部从头生成；pending 客户端不会清空）"
    )
    p_run.add_argument("--song", required=True)
    p_run.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    p_run.add_argument(
        "--continue",
        dest="fresh",
        action="store_false",
        help="不清空，以现有译文为起点重跑（默认：清空旧结果，从头生成）",
    )
    p_run.add_argument(
        "--skip-done", action="store_true", help="不清空，只补没定稿的段"
    )
    p_run.add_argument(
        "--keep-history",
        action="store_true",
        help="清空时不删，把旧 run 移到 archive/<时间戳>/",
    )
    p_run.add_argument(
        "--only", action="append", metavar="SECTION", help="只跑指定段，可重复"
    )
    p_run.add_argument("--redo-analysis", action="store_true", help="重跑 A")
    p_run.add_argument("--dry-run", action="store_true", help="只渲染请求，不调用模型")
    p_run.add_argument(
        "--client",
        choices=["pending", "api"],
        default=os.environ.get("DUBBING_CLIENT") or "pending",
        help="pending=不发请求（默认）；api=走 DeepSeek 的 Anthropic 兼容端点",
    )
    p_run.add_argument("--model", default=None, help="默认取 DUBBING_MODEL 或 deepseek-flash")
    p_run.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="通用温度，默认 0.7（或 DUBBING_TEMPERATURE）",
    )
    p_run.add_argument(
        "--temperature-a", type=float, default=None, help="只覆盖 A（体会意境）的温度"
    )
    p_run.add_argument(
        "--temperature-b", type=float, default=None, help="只覆盖 B（写译词）的温度"
    )
    p_run.add_argument("--max-tokens", type=int, default=8000)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
