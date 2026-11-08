#!/usr/bin/env python3
"""把已有 run 目录里的 agent 名改成新名。

    A/  →  perceiver/
    B/  →  writer/
    C/  →  reviewer/

这是**纯改名**：旧 run 都在 ``workflow/`` 下，而 workflow 管线读的"最终文本"就是 ``writer/``，
所以映射是一对一的。``mimic_lookup`` / ``mimic_write`` 只存在于新管线，旧 run 里没有、也不需要补。

脚本只动 ``run-*`` 目录（``preview-*`` 是请求快照，不改），并且是幂等的：
目标目录已存在就跳过并计数，绝不覆盖。

    # 先看会改什么
    python scripts/migrate_run_agents.py --dry-run --all

    # 真改（workflow/ 下的所有 run）
    python scripts/migrate_run_agents.py --all

    # 连 baselines/one-shot 一起改（默认不动 baseline）
    python scripts/migrate_run_agents.py --all --include-baselines
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.song import list_songs  # noqa: E402

__all__ = ["MAPPING", "plan_moves", "migrate_run", "main"]

MAPPING = {"A": "perceiver", "B": "writer", "C": "reviewer"}
BASELINE_DIR = Path("baselines") / "one-shot"


def iter_run_dirs(base: Path):
    if not base.is_dir():
        return
    for path in sorted(base.iterdir()):
        if path.is_dir() and path.name.startswith("run-"):
            yield path


def plan_moves(run_dir: Path) -> list[tuple[Path, Path]]:
    """列出这次要改的目录：整曲层的 A，和 ``sections/NNN/`` 下的 B / C。"""
    moves: list[tuple[Path, Path]] = []
    for old, new in MAPPING.items():
        src = run_dir / old
        if src.is_dir():
            moves.append((src, run_dir / new))

    sections = run_dir / "sections"
    if sections.is_dir():
        for section in sorted(p for p in sections.iterdir() if p.is_dir()):
            for old, new in MAPPING.items():
                src = section / old
                if src.is_dir():
                    moves.append((src, section / new))
    return moves


def _retag(directory: Path, agent: str) -> None:
    """``meta.json`` 里的 ``agent`` 字段跟着改名，别留下自相矛盾的记录。"""
    meta = directory / "meta.json"
    if not meta.is_file():
        return
    try:
        payload = json.loads(meta.read_text(encoding="utf-8"))
    except ValueError:
        return
    if not isinstance(payload, dict) or payload.get("agent") == agent:
        return
    payload["agent"] = agent
    meta.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def migrate_run(run_dir: Path, *, dry_run: bool = False, quiet: bool = False) -> tuple[int, int]:
    """返回 ``(改了几个, 跳过几个)``。"""
    moved = skipped = 0
    for src, dst in plan_moves(run_dir):
        if dst.exists():
            skipped += 1
            if not quiet:
                print(f"  跳过（{dst.name} 已存在）：{src}")
            continue
        if not quiet:
            print(f"  {src.relative_to(run_dir)} → {dst.name}")
        if dry_run:
            moved += 1
            continue
        src.rename(dst)
        _retag(dst, dst.name)
        moved += 1
    return moved, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="把 run 目录里的 A/B/C 改成 perceiver/writer/reviewer")
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "songs"), help="歌库目录")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--song", action="append", help="歌 id，可重复")
    targets.add_argument("--all", action="store_true", dest="all_songs", help="歌库里所有歌")
    parser.add_argument("--dry-run", action="store_true", help="只打印会改什么")
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="连 baselines/one-shot 的 run 一起改（默认不动 baseline）",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"找不到歌库目录 {root}", file=sys.stderr)
        return 1

    song_ids = list_songs(root=root) if args.all_songs else list(dict.fromkeys(args.song or []))
    if not song_ids:
        print("歌库是空的。", file=sys.stderr)
        return 1

    total_moved = total_skipped = total_runs = 0
    touched_songs: list[str] = []
    for song_id in song_ids:
        song_dir = root / song_id
        bases = [song_dir / "workflow"]
        if args.include_baselines:
            bases.append(song_dir / BASELINE_DIR)
        for base in bases:
            runs = list(iter_run_dirs(base))
            if not runs:
                continue
            if not args.quiet:
                print(f"{song_id} / {base.relative_to(root)}（{len(runs)} 个 run）")
            song_touched = False
            for run_dir in runs:
                moved, skipped = migrate_run(run_dir, dry_run=args.dry_run, quiet=args.quiet)
                total_runs += 1
                total_moved += moved
                total_skipped += skipped
                song_touched = song_touched or bool(moved or skipped)
            if song_touched and song_id not in touched_songs:
                touched_songs.append(song_id)

    verb = "将改" if args.dry_run else "已改"
    print(
        f"\n{verb} {total_moved} 个目录，跳过 {total_skipped} 个"
        f"（涉及 {len(touched_songs)} 首歌、{total_runs} 个 run）"
    )
    if args.dry_run:
        print("去掉 --dry-run 才会真的改。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
