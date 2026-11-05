#!/usr/bin/env python3
"""baseline：一句话提示词 + 一次性调用，作为多 agent 流水线的对照组。

做法很简单——把 `baseline/prompt.txt`（结尾是"歌词："）和歌曲数据接起来，
调一次模型，把模型原话写进 `data/songs/<歌名>/baseline.txt`。

数据格式（和 prompt.txt 里的说明一致）：

    言伝                              ← 第一行是标题
    いつか見た風景 6                    ← 日文原句 + 该句总音节数
    闇に日が差して手を伸ばしたい 12 6 6   ← 有 pattern 时再接分段音节数

`keep_source` 的乐句（保留原文，没有 budget）会被跳过——baseline 不需要知道
"这句不翻译"这种信息。

接口地址 / key / 模型都从 `.env` 读（`DUBBING_BASE_URL`、`DEEPSEEK_API_KEY`、
`DUBBING_MODEL`），和主流程共用一套配置。

用法：

    python baseline/run_baseline.py --song kotozute
    python baseline/run_baseline.py --all
    python baseline/run_baseline.py --song kotozute --dry-run   # 只看拼出来的输入
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.llm import APIClient, LLMError, load_dotenv  # noqa: E402
from dubbing.models import Song  # noqa: E402
from dubbing.song import list_songs, load_song  # noqa: E402

PROMPT_PATH = Path(__file__).resolve().parent / "prompt.txt"
OUT_NAME = "baseline.txt"
#: 整首歌一次吐完，思考块可能很长；给足预算（端点上限 64000）
BASELINE_MAX_TOKENS = 32000


# --------------------------------------------------------------------------
# 拼输入
# --------------------------------------------------------------------------


def lyrics_block(song: Song) -> tuple[str, list[str]]:
    """标题 + 每个乐句一行。返回 (文本, 被跳过的乐句 id)。"""
    lines = [song.title]
    skipped: list[str] = []

    for p in song.phrases:
        if p.budget is None:
            skipped.append(p.id)
            continue
        total = p.budget.max
        if p.pattern is not None:
            lines.append(f"{p.ja} {total} " + " ".join(str(n) for n in p.pattern))
        else:
            lines.append(f"{p.ja} {total}")

    return "\n".join(lines), skipped


def compose(song: Song) -> tuple[str, str]:
    """返回 (system, user)。

    ``system`` 是 prompt.txt 全文（结尾正好是"歌词："），``user`` 是歌词数据。
    拼起来与"把两段直接接成一整段"完全等价，只是分了两条消息。
    """
    prompt = PROMPT_PATH.read_text(encoding="utf-8").rstrip("\n")
    block, _ = lyrics_block(song)
    return prompt, block


def suspicious_lines(text: str) -> list[str]:
    """挑出看起来还带着序号/音节数的行（只提示，不改写输出）。"""
    bad = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if re.match(r"^\s*\d+[.、)．]", line) or re.search(r"[（(]\s*\d+", line) or re.search(r"\d", line):
            bad.append(line)
    return bad


# --------------------------------------------------------------------------
# 跑
# --------------------------------------------------------------------------


def run_one(
    song_id: str,
    *,
    root: Path,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int = BASELINE_MAX_TOKENS,
    dry_run: bool = False,
) -> Path:
    song = load_song(song_id, root=root)
    system, user = compose(song)
    _, skipped = lyrics_block(song)

    out_path = root / song_id / OUT_NAME
    print(f"\n=== {song_id}（{song.title}）===", file=sys.stderr)
    print(f"输入：{len(song.phrases)} 句，{len(skipped)} 句保留原文被跳过", file=sys.stderr)
    if skipped:
        print(f"  跳过：{', '.join(skipped)}", file=sys.stderr)

    if dry_run:
        print("--- system ---\n" + system, file=sys.stderr)
        print("--- user ---\n" + user, file=sys.stderr)
        return out_path

    client = APIClient(
        model=model,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = client.complete(agent="baseline", system=system, user=user)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected = len([p for p in song.phrases if p.budget is not None])
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) == expected + 1:
        # 模型偶尔还是会把标题翻出来当第一行，按行数判定并丢掉
        print(f"  丢弃首行（标题）：{lines[0][:40]}", file=sys.stderr)
        lines = lines[1:]
    elif len(lines) != expected:
        print(
            f"  注意：输出 {len(lines)} 行，预期 {expected} 行，逐行对齐可能有偏差",
            file=sys.stderr,
        )
    text = "\n".join(lines)

    out_path.write_text(text.rstrip() + "\n", encoding="utf-8")

    suspicious = suspicious_lines(text)
    if suspicious:
        print(
            f"  注意：有 {len(suspicious)} 行仍带序号或数字，输出格式可能没遵守",
            file=sys.stderr,
        )
        for line in suspicious[:3]:
            print(f"    {line[:60]}", file=sys.stderr)

    usage = client.last_usage or {}
    print(
        f"已写出 {out_path}（输出 {usage.get('output_tokens', '?')} tokens，{len(text)} 字）",
        file=sys.stderr,
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="baseline：单提示词一次调用")
    parser.add_argument("--root", default=str(REPO_ROOT / "data" / "songs"))
    parser.add_argument("--song", action="append", default=None, help="歌 id，可重复")
    parser.add_argument("--all", action="store_true", dest="all_songs")
    parser.add_argument("--dry-run", action="store_true", help="只打印拼好的输入")
    parser.add_argument("--model", default=None, help="默认取 DUBBING_MODEL")
    parser.add_argument("--base-url", default=None, help="默认取 DUBBING_BASE_URL")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=BASELINE_MAX_TOKENS)
    args = parser.parse_args(argv)

    root = Path(args.root)
    if args.all_songs:
        targets = list_songs(root=root)
    elif args.song:
        targets = list(args.song)
    else:
        print("要么给 --song <id>，要么加 --all。", file=sys.stderr)
        return 2
    if not targets:
        print(f"{root} 下没有歌", file=sys.stderr)
        return 1

    failures: list[tuple[str, str]] = []
    for song_id in targets:
        try:
            run_one(
                song_id,
                root=root,
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                dry_run=args.dry_run,
            )
        except (LLMError, FileNotFoundError, ValueError) as exc:
            print(f"  ✗ {song_id}: {exc}", file=sys.stderr)
            failures.append((song_id, str(exc)))

    if len(targets) > 1:
        print(
            f"\n结束：{len(targets) - len(failures)} 首成功 / {len(failures)} 首失败",
            file=sys.stderr,
        )
        for song_id, reason in failures:
            print(f"  ✗ {song_id}: {reason[:80]}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
