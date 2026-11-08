#!/usr/bin/env python3
"""统一的 baseline 入口：一个脚本跑三种 baseline。

| 名字 | 提示词 | 模型看到的上下文 | 架构 |
| --- | --- | --- | --- |
| ``one-shot`` | workflow 的 ``shared.md`` + ``b.md`` | 整首歌、已译好的其他段、budget / pattern / keep_source | **只有 B 一次调用**，没有 A 分析、没有 C 检查与返修 |
| ``paper-original`` | 论文 Exp.4 原指令 | 本段原文逐行；不传预算、分组、保留标记、标题 | 生成 + 同模型清理格式 |
| ``paper-annotated`` | 论文 Exp.4 的人工预算适配版 | 本段 ``id`` / ``ja`` / ``budget`` / ``pattern`` / ``keep_source`` | 生成 + 同模型清理格式 |

``one-shot`` 是 workflow 的消融对照：workflow 给的输入它都给（按段推进，翻译当前段时
看得到整首歌词和已经定稿的部分），差别只在没有 A 的意境分析和 C 的“检查 → 返修”闭环。

三者的输出组织与 workflow 完全一致：

    data/songs/<song_id>/baselines/<名字>/<run_id>/
      latest.json                      # 在 <名字>/ 下，指向当前 run
      sections/001/writer/{request,output,meta}.json        # one-shot
      sections/001/generation/{request,output,meta}.json   # paper-*
      sections/001/cleanup/{request,output,meta}.json
      lyrics.zh.txt
      lyrics.md

用法：

    python baseline/run_baselines.py --song kotozute --baseline one-shot --dry-run
    python baseline/run_baselines.py --all --baseline paper --client api
    python baseline/run_baselines.py --song kotozute --baseline one-shot --client api
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.agents import build_writer_request, parse_draft_output  # noqa: E402
from dubbing.checks import review  # noqa: E402
from dubbing.export import write_lyrics  # noqa: E402
from dubbing.llm import (  # noqa: E402
    APIClient,
    DEFAULT_MAX_TOKENS,
    LLMClient,
    LLMError,
    extract_json,
    load_dotenv,
)
from dubbing.models import Analysis, Draft, DraftPhrase, Song, View, ViewRow  # noqa: E402
from dubbing.prompting import prompt_hash, system_for  # noqa: E402
from dubbing.song import list_songs, load_song  # noqa: E402
from dubbing.store import Request, RunStore, run_base  # noqa: E402

__all__ = [
    "VARIANTS",
    "BaselineOutputError",
    "build_paper_request",
    "build_one_shot_request",
    "run_paper",
    "run_one_shot",
    "evaluate_view",
    "main",
]

PROMPTS = Path(__file__).resolve().parent
VARIANTS = ("one-shot", "paper-original", "paper-annotated")
#: ``--baseline paper`` 展开成这两组。
PAPER_VARIANTS = ("paper-original", "paper-annotated")
PAPER_URL = "https://aclanthology.org/2026.eacl-srw.42/"


class BaselineOutputError(ValueError):
    """清理输出不能可靠地与输入行一一对齐。"""


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _base(root: Path | str, song_id: str, baseline: str) -> Path:
    return run_base(root, song_id, f"baselines/{baseline}")


# --------------------------------------------------------------------------
# 请求构造
# --------------------------------------------------------------------------


def build_one_shot_request(song: Song, section: str, accepted: dict[str, str]) -> Request:
    """workflow 的 B 提示词 + 全曲上下文，但没有 A 分析、没有上一轮反馈。"""
    analysis = Analysis(theme="", emotional_arc="", narrative="")
    return build_writer_request(song, analysis, focus=section, accepted=accepted, prev_texts=None)


def build_paper_request(song: Song, section: str, variant: str) -> Request:
    """原样组的模型输入不含人工预算、分组、保留标记或提示指令。"""
    if variant not in PAPER_VARIANTS:
        raise ValueError(f"未知 paper baseline：{variant}")
    song.section(section)
    phrases = song.phrases_of(section)
    if variant == "paper-original":
        system = _prompt("paper_original.txt").format(target_language="Chinese")
        data = {"lyrics": [p.ja for p in phrases]}
        user = "\n".join(data["lyrics"])
    else:
        system = _prompt("paper_annotated.txt").format(target_language="Chinese")
        rows = []
        for p in phrases:
            row: dict[str, Any] = {
                "id": p.id,
                "ja": p.ja,
                "budget": p.budget.to_json() if p.budget is not None else None,
            }
            if p.pattern is not None:
                row["pattern"] = list(p.pattern)
            if p.keep_source:
                row["keep_source"] = True
            rows.append(row)
        data = {"phrases": rows}
        user = _json(data)
    return Request(system=system, user=user, data=data)


def cleanup_request(response: str) -> Request:
    """两组 paper 使用相同的格式清理，不传原文或人工预算。"""
    return Request(system=_prompt("paper_cleanup.txt"), user=response, data={"response": response})


def parse_lines(response: str, expected: int) -> list[str]:
    try:
        payload = extract_json(response)
    except LLMError as exc:
        raise BaselineOutputError(f"无法解析格式清理结果：{exc}") from exc
    lines = payload.get("lines") if isinstance(payload, dict) else None
    if not isinstance(lines, list) or any(not isinstance(line, str) or not line.strip() for line in lines):
        raise BaselineOutputError("清理输出必须为含非空字符串 lines 数组的 JSON 对象")
    if len(lines) != expected:
        raise BaselineOutputError(f"清理输出 {len(lines)} 行，预期 {expected} 行；未自动删除或补写任何行")
    return [line.strip() for line in lines]


def align_lines(generated: str, cleaned: str, phrase_ids: list[str]) -> tuple[list[str], str]:
    """结构化译文直接按 ID 取原词；格式清理不能抹去或改写已知 zh。"""
    try:
        payload = extract_json(generated)
    except LLMError:
        payload = None
    if isinstance(payload, dict) and isinstance(payload.get("phrases"), list):
        rows = payload["phrases"]
        if rows and all(isinstance(row, dict) and "id" in row and "zh" in row for row in rows):
            ids = [row["id"] for row in rows]
            if any(not isinstance(pid, str) for pid in ids) or len(set(ids)) != len(ids) or set(ids) != set(phrase_ids):
                raise BaselineOutputError("结构化译文的 phrase ID 重复、缺失或未知，无法可靠对齐")
            texts = {row["id"]: row["zh"] for row in rows}
            if any(not isinstance(text, str) or not text.strip() for text in texts.values()):
                raise BaselineOutputError("结构化译文包含空白或非字符串 zh")
            return [texts[pid].strip() for pid in phrase_ids], "generation.phrases.zh"
    return parse_lines(cleaned, len(phrase_ids)), "cleanup.lines"


class PaperClient(APIClient):
    """沿用现有接口、重试和采样配置，额外留存完整 JSON 响应供核对。"""

    def __init__(self, *args, protocol: str = "anthropic", thinking: str | None = None, **kwargs) -> None:
        if protocol not in ("anthropic", "openai"):
            raise ValueError(f"未知接口协议：{protocol}")
        self.protocol = protocol
        if thinking not in (None, "enabled", "disabled"):
            raise ValueError(f"未知思考设置：{thinking}")
        self.thinking = thinking
        super().__init__(*args, **kwargs)

    @property
    def url(self) -> str:
        if self.protocol == "openai":
            return f"{self.base_url}/chat/completions"
        return super().url

    def _headers(self) -> dict[str, str]:
        if self.protocol == "openai":
            if not self.api_key:
                raise LLMError("没有 API key：请设置 DEEPSEEK_API_KEY")
            return {"content-type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        return super()._headers()

    def _body(self, system: str, user: str, *, agent: str) -> dict[str, Any]:
        if self.protocol == "openai":
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature_for(agent),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if self.thinking is not None:
                body["enable_thinking"] = self.thinking == "enabled"
            return body
        body = super()._body(system, user, agent=agent)
        if self.thinking is not None:
            body["thinking"] = {"type": self.thinking}
        return body

    def complete(self, *, agent: str, system: str, user: str) -> str:
        self.last_response = None
        self.last_usage = None
        self.last_thinking_observation = None
        return super().complete(agent=agent, system=system, user=user)

    def _text_of(self, resp, *, agent: str) -> str:
        try:
            self.last_response = resp.json()
        except ValueError:
            pass
        data = self.last_response
        if self.protocol == "openai":
            if not isinstance(data, dict):
                raise LLMError(f"agent={agent} 返回的不是 JSON 对象")
            if "error" in data:
                raise LLMError(f"agent={agent} 调用失败：{str(data['error'])[:300]}")
            self.last_usage = data.get("usage")
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LLMError(f"agent={agent} 的响应里没有 choices")
            choice = choices[0]
            message = choice.get("message")
            usage = self.last_usage or {}
            details = usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens")
            has_reasoning = isinstance(message, dict) and bool(
                message.get("reasoning_content") or message.get("reasoning")
            )
            self.last_thinking_observation = {
                "requested": self.thinking,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_content_present": has_reasoning,
            }
            if self.thinking == "disabled" and (
                has_reasoning or isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0
            ):
                raise LLMError("已请求 enable_thinking=false，但网关仍返回推理内容或 reasoning tokens；停止该任务")
            if choice.get("finish_reason") == "length":
                raise LLMError(f"agent={agent} 的输出被 max_tokens={self.max_tokens} 截断")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise LLMError(f"agent={agent} 的响应里没有正文，finish_reason={choice.get('finish_reason')}")
            return content.strip()
        if isinstance(data, dict):
            self.last_usage = data.get("usage")
            blocks = data.get("content")
            blocks = blocks if isinstance(blocks, list) else []
            has_reasoning = any(
                isinstance(block, dict)
                and (block.get("type") == "redacted_thinking" or block.get("type") == "thinking" and bool(block.get("thinking")))
                for block in blocks
            )
            usage = self.last_usage or {}
            reasoning_tokens = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
            self.last_thinking_observation = {
                "requested": self.thinking,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_content_present": has_reasoning,
            }
            if self.thinking == "disabled" and (
                has_reasoning or isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0
            ):
                raise LLMError("已请求 thinking.type=disabled，但接口仍返回推理内容或 reasoning tokens；停止该任务")
        return super()._text_of(resp, agent=agent)


# --------------------------------------------------------------------------
# 共用：调用 + 评测 + 导出
# --------------------------------------------------------------------------


def _meta(agent: str, client: LLMClient, system: str, **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "model": getattr(client, "model", client.name),
        "client": client.name,
        "prompt_hash": prompt_hash(system),
        **extra,
    }
    for key, attr in (
        ("base_url", "base_url"),
        ("protocol", "protocol"),
        ("usage", "last_usage"),
        ("temperature", "last_temperature"),
        ("thinking_observation", "last_thinking_observation"),
    ):
        value = getattr(client, attr, None)
        if value is not None:
            meta[key] = value
    return meta


def _call(
    store: RunStore,
    run_id: str,
    agent: str,
    request: Request,
    client: LLMClient,
    *,
    index: int,
    llm_agent: str,
    extra_meta: dict[str, Any] | None = None,
) -> str:
    """写 request.json → 调一次模型 → 返回原始正文（output.json 由调用方决定）。"""
    store.prepare(run_id, agent, request, section_index_=index)
    try:
        return client.complete(agent=llm_agent, system=request.system, user=request.user)
    except LLMError as exc:
        store.abandon(
            run_id, agent, reason=str(exc), meta=_meta(agent, client, request.system, **(extra_meta or {})),
            section_index_=index,
        )
        raise


def evaluate_view(song: Song, texts: dict[str, str], run_id: str) -> View:
    """统一用人工预算事后评测；规则结论只进视图，不反馈给任何模型。"""
    reviews = {}
    for section in song.sections:
        draft = Draft(
            section=section.id,
            phrases=tuple(DraftPhrase(id=p.id, zh=texts.get(p.id, "")) for p in song.translatable_of(section.id)),
        )
        reviews[section.id] = review(song, section.id, draft, draft_run=run_id)
    rows = []
    for p in song.phrases:
        text = p.ja if p.keep_source else texts.get(p.id, "")
        verdict = "skip" if p.keep_source else ("fail" if p.id in reviews[p.section].failing_ids() else "pass")
        rows.append(
            ViewRow(
                phrase_id=p.id,
                section=p.section,
                ja=p.ja,
                budget=p.budget,
                lang="ja" if p.keep_source else ("zh" if text else None),
                sing_text=text,
                verdict=verdict,
                draft_run=run_id,
            )
        )
    return View(song_id=song.id, rows=tuple(rows))


def _finish_run(store: RunStore, song: Song, run_id: str, texts: dict[str, str], client: LLMClient) -> list[Path]:
    view = evaluate_view(song, texts, run_id)
    paths = write_lyrics(song, view, store.run_dir(run_id))
    store.set_latest(
        run_id,
        song=song.id,
        status="completed",
        model=getattr(client, "model", client.name),
        passed=sum(1 for r in view.rows if r.verdict == "pass"),
        total=len(view.rows),
    )
    return paths


# --------------------------------------------------------------------------
# one-shot：workflow 提示词 + 单次调用（消融对照）
# --------------------------------------------------------------------------


def run_one_shot(song: Song, client: LLMClient, *, out: Path, dry_run: bool = False) -> Path:
    store = RunStore(out)
    run_id = store.new_preview() if dry_run else store.new_run()
    texts: dict[str, str] = {}

    for index, section in enumerate(song.sections, 1):
        if not song.translatable_of(section.id):
            continue
        request = build_one_shot_request(song, section.id, texts)
        if dry_run:
            store.prepare(run_id, "writer", request, section_index_=index)
            continue

        print(f"[one-shot] {song.id} / {section.id} 生成", file=sys.stderr)
        raw = _call(
            store, run_id, "writer", request, client,
            index=index, llm_agent="writer", extra_meta={"section": section.id},
        )
        try:
            draft, warnings = parse_draft_output(extract_json(raw), song, section.id)
        except (LLMError, ValueError) as exc:
            store.abandon(
                run_id, "writer", reason=f"解析失败：{exc}", raw_response=raw,
                meta=_meta("writer", client, request.system, section=section.id), section_index_=index,
            )
            raise
        store.finish(
            run_id, "writer", draft.to_json(),
            meta=_meta("writer", client, request.system, section=section.id, warnings=warnings),
            raw_response=raw, section_index_=index,
        )
        texts.update(draft.texts())

    if dry_run:
        return store.run_dir(run_id)
    _finish_run(store, song, run_id, texts, client)
    return store.run_dir(run_id)


# --------------------------------------------------------------------------
# paper-original / paper-annotated
# --------------------------------------------------------------------------


def run_paper(song: Song, variant: str, client: LLMClient, *, out: Path, dry_run: bool = False) -> Path:
    store = RunStore(out)
    run_id = store.new_preview() if dry_run else store.new_run()
    texts: dict[str, str] = {}

    for index, section in enumerate(song.sections, 1):
        request = build_paper_request(song, section.id, variant)
        if dry_run:
            store.prepare(run_id, "generation", request, section_index_=index)
            continue

        print(f"[{variant}] {song.id} / {section.id} 生成", file=sys.stderr)
        raw = _call(
            store, run_id, "generation", request, client,
            index=index, llm_agent="baseline", extra_meta={"section": section.id, "variant": variant},
        )
        gen_meta = _meta("generation", client, request.system, section=section.id, variant=variant)
        # 先把原始正文落盘，后面清理或对齐失败也不会丢证据
        store.finish(
            run_id, "generation", {"section": section.id, "source": None, "phrases": []},
            meta=gen_meta, raw_response=raw, section_index_=index,
        )

        clean_req = cleanup_request(raw)
        cleaned = _call(
            store, run_id, "cleanup", clean_req, client,
            index=index, llm_agent="baseline", extra_meta={"section": section.id, "variant": variant},
        )
        phrases = song.phrases_of(section.id)
        try:
            lines, source = align_lines(raw, cleaned, [p.id for p in phrases])
        except (BaselineOutputError, LLMError) as exc:
            store.abandon(
                run_id, "cleanup", reason=str(exc), raw_response=cleaned,
                meta=_meta("cleanup", client, clean_req.system, section=section.id, variant=variant),
                section_index_=index,
            )
            raise BaselineOutputError(f"{song.id}/{section.id}: {exc}") from exc

        store.finish(
            run_id, "cleanup", {"section": section.id, "lines": _cleanup_lines(cleaned)},
            meta=_meta("cleanup", client, clean_req.system, section=section.id, variant=variant),
            raw_response=cleaned, section_index_=index,
        )
        store.finish(
            run_id, "generation",
            {"section": section.id, "source": source, "phrases": [
                {"id": p.id, "zh": line} for p, line in zip(phrases, lines)
            ]},
            meta=gen_meta, raw_response=raw, section_index_=index,
        )
        texts.update((p.id, line) for p, line in zip(phrases, lines))

    if dry_run:
        return store.run_dir(run_id)
    _finish_run(store, song, run_id, texts, client)
    return store.run_dir(run_id)


def _cleanup_lines(cleaned: str) -> list[str]:
    try:
        payload = extract_json(cleaned)
    except LLMError:
        return []
    lines = payload.get("lines") if isinstance(payload, dict) else None
    return [line for line in lines if isinstance(line, str)] if isinstance(lines, list) else []


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _expand(raw: list[str]) -> list[str]:
    out: list[str] = []
    for name in raw:
        if name == "paper":
            out.extend(PAPER_VARIANTS)
        elif name == "all":
            out.extend(VARIANTS)
        else:
            out.append(name)
    return list(dict.fromkeys(out))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="统一的 baseline 入口：one-shot / paper-original / paper-annotated")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "songs")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--song", action="append", help="歌曲 ID，可重复")
    targets.add_argument("--all", action="store_true", dest="all_songs")
    parser.add_argument(
        "--baseline",
        action="append",
        choices=(*VARIANTS, "paper", "all"),
        default=None,
        help="可重复；paper=两组论文 baseline，all=全部三种（默认）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只保存请求，不调用模型")
    parser.add_argument("--client", choices=("pending", "api"), default="pending", help="真实调用须显式选择 api")
    parser.add_argument("--model", default=None, help="默认使用 DUBBING_MODEL 或项目默认模型")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--protocol", choices=("anthropic", "openai"), default="anthropic", help="openai 使用 /chat/completions 和 Bearer 认证")
    parser.add_argument("--thinking", choices=("none", "enabled", "disabled"), default="none", help="默认不发思考开关，三组条件一致")
    parser.add_argument("--max-retries", type=int, default=3, help="每次接口调用的最大尝试次数，默认 3")
    parser.add_argument("--jobs", type=int, default=1, help="并发的歌曲 / baseline 任务数，默认 1；每个任务独立客户端")
    parser.add_argument("--temperature", type=float, default=None, help="默认使用当前项目配置；三组一致")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens 必须为正整数")
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")
    if args.max_retries < 1:
        parser.error("--max-retries 必须为正整数")
    if not args.dry_run and args.client != "api":
        print("需要 --client api 才会运行；可先使用 --dry-run 查看请求。", file=sys.stderr)
        return 2

    variants = _expand(args.baseline or ["all"])
    song_ids = list_songs(root=args.root) if args.all_songs else list(dict.fromkeys(args.song))
    if not song_ids:
        print("歌库为空。", file=sys.stderr)
        return 1

    failures: list[str] = []
    tasks = []
    for song_id in song_ids:
        try:
            song = load_song(song_id, root=args.root)
        except (OSError, ValueError) as exc:
            print(f"{song_id}: {exc}", file=sys.stderr)
            failures.append(song_id)
            continue
        tasks.extend((song_id, song, variant) for variant in variants)

    def execute(task):
        song_id, song, variant = task
        client = PaperClient(
            model=args.model, base_url=args.base_url, temperature=args.temperature,
            max_tokens=args.max_tokens, protocol=args.protocol,
            thinking=None if args.thinking == "none" else args.thinking,
            max_retries=args.max_retries,
        )
        out = _base(args.root, song_id, variant)
        if variant == "one-shot":
            return variant, run_one_shot(song, client, out=out, dry_run=args.dry_run)
        return variant, run_paper(song, variant, client, out=out, dry_run=args.dry_run)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(pending):
            song_id, song, variant = pending[future]
            try:
                _variant, directory = future.result()
                print(f"{song_id} / {variant} -> {directory}", flush=True)
            except (OSError, ValueError, LLMError) as exc:
                failures.append(f"{song_id}/{variant}")
                print(f"{song_id} / {variant}: {exc}", file=sys.stderr)

    print(f"完成：{len(song_ids)} 首，{len(variants)} 组；失败 {len(failures)} 项。")
    if failures:
        for item in failures:
            print(f"  ✗ {item}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
