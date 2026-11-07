"""命令行入口。

    python -m dubbing songs
    python -m dubbing run   --song kotozute
    python -m dubbing run   --song kotozute --dry-run
    python -m dubbing view  --song kotozute

一次 ``run`` 在 ``data/songs/<id>/workflow/`` 下产生一个 run 目录；旧 run 不会被
覆盖，``latest.json`` 决定 ``view`` / ``export`` 默认看哪一次。
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
    DEFAULT_ROOT,
    Step,
    compose_request,
    plan_requests,
    run_song,
    workflow_store,
)
from .report import list_runs, render_record, table
from .song import list_songs, load_song
from .store import section_index
from .view import build_view

__all__ = ["main"]


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
        store = workflow_store(Path(args.root), sid)
        run_id = store.latest_run()
        runs = len(store.list_runs())
        print(
            f"{sid:<12} {song.title:<10} {len(song.sections)} 段 / {n} 句 / 预算共 {total} 字"
            f" / workflow {runs} run"
        )
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


def _resolve_run(store, args, song) -> str | None:
    """``--run`` 优先，否则用 ``latest.json`` 指的那一次。"""
    if getattr(args, "run", None):
        return args.run
    return store.latest_run()


def cmd_view(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = workflow_store(Path(args.root), args.song)
    run_id = _resolve_run(store, args, song)
    view = build_view(song, store, run_id)

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
    print(f"run：{run_id or '（还没有 run）'}")
    print(f"看成品歌词：python -m dubbing export --song {args.song}")
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
    store = workflow_store(Path(args.root), args.song)
    rows = list_runs(store, song)
    if not rows:
        print("还没有任何 run。先跑一次：python -m dubbing run --song <id> --client api")
        return 0

    out = [["", "run_id", "模型", "A", "段落", "歌词", "tokens", "状态"]]
    for r in rows:
        out.append(
            [
                "*" if r["current"] else "",
                r["run_id"],
                r["model"],
                "有" if r["has_a"] else "—",
                f"{r['passed']}/{r['total']} pass",
                "有" if r["lyrics"] else "—",
                "" if r["out_tokens"] is None else str(r["out_tokens"]),
                r["status"],
            ]
        )
    print(f"{song.title}（{args.song}）")
    print(table(out))
    print("\n* = latest.json 当前指向的一次")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = workflow_store(Path(args.root), args.song)
    run_id = _resolve_run(store, args, song)

    if not run_id:
        print("还没有任何 run。先跑一次：python -m dubbing run --song <id> --client api")
        return 0

    sections = [args.section] if args.section else None
    if sections:
        missing = [s for s in sections if s not in song.section_ids]
        if missing:
            print(f"没有这些 section：{missing}", file=sys.stderr)
            return 1

    printed = False
    if args.agent in (None, "a"):
        print(render_record(song, store, run_id, "a", None), end="")
        printed = True
    if args.agent != "a":
        targets = sections or list(song.section_ids)
        for sid in targets:
            if args.agent in (None, "b"):
                print(render_record(song, store, run_id, "b", sid), end="")
                printed = True
            if args.agent in (None, "c"):
                print(render_record(song, store, run_id, "c", sid), end="")
                printed = True

    if not printed:
        print("还没有产物。先跑一次：python -m dubbing run --song <id> --client api")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    song = load_song(args.song, root=Path(args.root))
    store = workflow_store(Path(args.root), args.song)

    if args.run:
        agent = (args.agent or "b").upper()
        section = args.section
        if agent != "A" and section is None:
            section = next((s for s in song.section_ids if song.translatable_of(s)), None)
        index = section_index(song, section) if (agent != "A" and section) else None
        payload = store.read_request(args.run, agent, index)
        meta = store.read_meta(args.run, agent, index)
        system = payload["messages"]["system"]
        user = payload["messages"]["user"]
        where = f"，section {section}" if section else ""
        head = f"# {args.run}（agent {agent}{where}）真实发出去的请求"
        if meta.get("model"):
            head += f"，model {meta['model']}"
    else:
        request = compose_request(
            args.song, agent=args.agent or "b", section=args.section, root=Path(args.root)
        )
        system, user = request.system, request.user
        where = f"，section {args.section}" if args.section else "，section 取第一个可译段"
        head = f"# {song.id} 现拼的请求（agent {(args.agent or 'b').upper()}{where}）"

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
    store = workflow_store(Path(args.root), args.song)
    run_id = _resolve_run(store, args, song)
    if not run_id:
        print("还没有任何 run，没有可导出的歌词。", file=sys.stderr)
        return 1
    view = build_view(song, store, run_id)

    out_dir = Path(args.out) if args.out else store.run_dir(run_id)
    paths = write_lyrics(song, view, out_dir)
    for p in paths:
        print(p)
    print()
    print(paths[0].read_text(encoding="utf-8"))
    return 0


def _run_one(song_id: str, args: argparse.Namespace, root: Path) -> int:

    if args.dry_run:
        plan = plan_requests(
            song_id, root=root, skip_done=args.skip_done, fresh=args.fresh
        )
        song = load_song(song_id, root=root)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        out_dir = workflow_store(root, song_id).base / f"preview-{stamp}"
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
        if any(name == "A" for name, _ in plan):
            print("注意：A 还没跑，B 的 analysis 会是空的；先跑通 A 才有意义。")
        print(f"共 {len(song.phrases)} 句，{len(plan)} 个请求。")
        return 0

    llm = build_client(
        args.client,
        model=args.model,
        base_url=args.base_url,
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
            song_id,
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
        print("查一遍 DEEPSEEK_API_KEY 和 --model；入参在 workflow/<run>/ 里。", file=sys.stderr)
        return 1
    except SchemaError as exc:
        print(f"数据格式有问题：{exc}", file=sys.stderr)
        return 1

    print(f"\n{song_id}  run {outcome.run_id}")
    print(f"  A: {'复用上一版' if outcome.analysis_reused else '新跑'}")
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
    p_view.set_defaults(func=cmd_view)

    p_runs = sub.add_parser("runs", help="列出所有 run（标出 latest.json 指的是哪一次）")
    p_runs.add_argument("--song", required=True)
    p_runs.set_defaults(func=cmd_runs)

    p_show = sub.add_parser("show", help="看某个 agent 的产物")
    p_show.add_argument("--song", required=True)
    p_show.add_argument("--agent", choices=["a", "b", "c"], default=None)
    p_show.add_argument("--section", default=None)
    p_show.add_argument("--run", default=None, help="指定 run，默认看 latest.json")
    p_show.set_defaults(func=cmd_show)

    p_prompt = sub.add_parser("prompt", help="看某个 agent 收到的提示词与数据")
    p_prompt.add_argument("--song", required=True)
    p_prompt.add_argument("--agent", choices=["a", "b"], default="b")
    p_prompt.add_argument("--section", default=None, help="不给就取第一个可译段")
    p_prompt.add_argument("--run", default=None, help="读某次真实调用的 request.json")
    p_prompt.set_defaults(func=cmd_prompt)

    p_export = sub.add_parser("export", help="导出歌词文件（lyrics.zh.txt / lyrics.md）")
    p_export.add_argument("--song", required=True)
    p_export.add_argument("--run", default=None, help="指定 run，默认看 latest.json")
    p_export.add_argument("--out", default=None, help="输出目录，默认该 run 目录")
    p_export.set_defaults(func=cmd_export)

    p_run = sub.add_parser(
        "run",
        help="跑一遍，在 workflow/ 下新建一个 run（A/B/C 全部从头生成）",
    )
    p_run.add_argument(
        "--song", action="append", default=None, help="歌 id；可重复，也可用 --all"
    )
    p_run.add_argument(
        "--all", action="store_true", dest="all_songs", help="跑歌库里所有歌"
    )
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
        "--temperature-a", type=float, default=None, help="只覆盖 A（体会意境）的温度"
    )
    p_run.add_argument(
        "--temperature-b", type=float, default=None, help="只覆盖 B（写译词）的温度"
    )
    p_run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
