#!/usr/bin/env python3
"""论文 Exp.4 双对照：原提示 + 原文，或人工 budget / pattern 适配版。

两组共用当前模型配置；按段独立生成，再各调用一次同模型清理格式。
C 仅在生成结束后评测，不将检查结果反馈给模型。现有 baseline.txt 不变。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.checks import review  # noqa: E402
from dubbing.counting import count_syllables  # noqa: E402
from dubbing.export import write_lyrics  # noqa: E402
from dubbing.llm import APIClient, DEFAULT_MAX_TOKENS, LLMClient, LLMError, extract_json, load_dotenv  # noqa: E402
from dubbing.models import Draft, DraftPhrase, Song, View, ViewRow  # noqa: E402
from dubbing.song import list_songs, load_song  # noqa: E402
from dubbing.store import Request  # noqa: E402

PROMPTS = Path(__file__).resolve().parent
VARIANTS = ("paper-original", "paper-annotated")
PAPER_URL = "https://aclanthology.org/2026.eacl-srw.42/"


class BaselineOutputError(ValueError):
    """清理输出不能可靠地与输入行一一对齐。"""


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(obj) + "\n", encoding="utf-8")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8").strip()


def build_request(song: Song, section: str, variant: str) -> Request:
    """原样组的模型输入不含人工预算、分组、保留标记或提示指令。"""
    if variant not in VARIANTS:
        raise ValueError(f"未知 baseline：{variant}")
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
            row = {
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
    """两组使用相同的格式清理，不传原文或人工预算，也不返修歌词。"""
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
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature_for(agent),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
            if self.thinking is not None:
                body["enable_thinking"] = self.thinking == "enabled"
        else:
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
        if self.protocol == "openai":
            data = self.last_response
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
            has_reasoning = isinstance(message, dict) and bool(message.get("reasoning_content") or message.get("reasoning"))
            self.last_thinking_observation = {"requested": self.thinking, "reasoning_tokens": reasoning_tokens, "reasoning_content_present": has_reasoning}
            if self.thinking == "disabled" and (has_reasoning or isinstance(reasoning_tokens, (int, float)) and reasoning_tokens > 0):
                raise LLMError("已请求 enable_thinking=false，但网关仍返回推理内容或 reasoning tokens；停止该任务以避免混入思考模式结果")
            if choice.get("finish_reason") == "length":
                raise LLMError(f"agent={agent} 的输出被 max_tokens={self.max_tokens} 截断")
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, str) or not content.strip():
                raise LLMError(f"agent={agent} 的响应里没有正文，finish_reason={choice.get('finish_reason')}")
            return content.strip()
        data = self.last_response
        if isinstance(data, dict):
            self.last_usage = data.get("usage")
            blocks = data.get("content")
            blocks = blocks if isinstance(blocks, list) else []
            has_reasoning = any(
                isinstance(block, dict) and (
                    block.get("type") == "redacted_thinking"
                    or block.get("type") == "thinking" and bool(block.get("thinking"))
                )
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
                raise LLMError("已请求 thinking.type=disabled，但接口仍返回推理内容或 reasoning tokens；停止该任务以避免混入思考模式结果")
        return super()._text_of(resp, agent=agent)


def _call(client: LLMClient, request: Request, directory: Path, *, reuse: bool = False) -> dict[str, Any]:
    payload = request.to_json()
    if isinstance(client, APIClient):
        payload["transport"] = {"url": client.url, "body": client._body(request.system, request.user, agent="baseline")}
    if reuse and (directory / "meta.json").exists():
        previous = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        saved_request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        if saved_request != payload:
            raise ValueError(f"已保存请求与当前参数不同，拒绝复用：{directory}")
        if previous.get("status") == "completed" and (directory / "response.txt").exists():
            return {"response": (directory / "response.txt").read_text(encoding="utf-8"), **previous, "reused": True}
        archive = directory / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8])
        archive.mkdir(parents=True)
        for path in directory.iterdir():
            if path.is_file():
                shutil.copy2(path, archive / path.name)
    _write(directory / "request.json", payload)
    started = time.monotonic()
    meta: dict[str, Any] = {"status": "running"}
    _write(directory / "meta.json", meta)
    try:
        response = client.complete(agent="baseline", system=request.system, user=request.user)
        (directory / "response.txt").write_text(response, encoding="utf-8")
        meta["status"] = "completed"
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        raise
    finally:
        meta["elapsed_seconds"] = round(time.monotonic() - started, 3)
        meta["usage"] = getattr(client, "last_usage", None)
        meta["thinking_observation"] = getattr(client, "last_thinking_observation", None)
        full_response = getattr(client, "last_response", None)
        if full_response is not None:
            _write(directory / "response.json", full_response)
        _write(directory / "meta.json", meta)
    return {"response": response, **meta}


def evaluate(song: Song, texts: dict[str, str], run_id: str) -> tuple[View, dict[str, Any]]:
    """统一用人工预算事后评测两组；检查结论不进入任何模型请求。"""
    reviews = {}
    for section in song.sections:
        draft = Draft(section=section.id, phrases=tuple(
            DraftPhrase(id=p.id, zh=texts[p.id]) for p in song.translatable_of(section.id)
        ))
        reviews[section.id] = review(song, section.id, draft, draft_run=run_id)
    rows = []
    details = []
    for p in song.phrases:
        text = p.ja if p.keep_source else texts[p.id]
        verdict = "skip" if p.keep_source else (
            "fail" if p.id in reviews[p.section].failing_ids() else "pass"
        )
        rows.append(ViewRow(
            phrase_id=p.id, section=p.section, ja=p.ja, budget=p.budget,
            lang="ja" if p.keep_source else "zh", sing_text=text, verdict=verdict,
        ))
        actual = None if p.keep_source else count_syllables(text)
        distance = None
        if p.budget is not None and actual is not None:
            distance = max(p.budget.min - actual, actual - p.budget.max, 0)
        details.append({
            "id": p.id, "section": p.section, "model_text": texts[p.id], "sing_text": text,
            "keep_source": p.keep_source, "actual": actual,
            "budget": p.budget.to_json() if p.budget is not None else None,
            "budget_distance": distance, "verdict": verdict,
        })
    evaluated = [d for d in details if d["budget_distance"] is not None]
    translatable = [d for d in details if not d["keep_source"]]
    report = {
        "evaluated_phrases": len(evaluated),
        "kept_phrases": len(details) - len(translatable),
        "budget_pass_rate": sum(d["budget_distance"] == 0 for d in evaluated) / len(evaluated) if evaluated else None,
        "mean_budget_distance": sum(d["budget_distance"] for d in evaluated) / len(evaluated) if evaluated else None,
        "rule_pass_rate": sum(d["verdict"] == "pass" for d in translatable) / len(translatable) if translatable else None,
        "phrases": details,
        "sections": [r.to_json() for r in reviews.values()],
        "note": "仅为人工预算和规则的事后评测，不是论文原始节奏距离，也不代表真人可唱性评分。",
    }
    return View(song_id=song.id, rows=tuple(rows)), report


def _code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_one(song: Song, variant: str, client: LLMClient, *, out: Path, dry_run: bool = False, resume_dir: Path | None = None) -> Path:
    requests = [(s.id, build_request(song, s.id, variant)) for s in song.sections]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = resume_dir or out / variant / (("preview-" if dry_run else "run-") + stamp + "-" + uuid4().hex[:8])
    snapshot = song.to_json()
    config = {
        "song_id": song.id, "variant": variant, "paper": PAPER_URL, "experiment": 4,
        "model": getattr(client, "model", client.name),
        "base_url": getattr(client, "base_url", None),
        "protocol": getattr(client, "protocol", None),
        "thinking_requested": getattr(client, "thinking", None),
        "temperature_requested": client.temperature_for("baseline") if isinstance(client, APIClient) else None,
        "top_p_requested": None,
        "max_tokens": getattr(client, "max_tokens", None),
        "max_retries": getattr(client, "max_retries", None),
        "code_commit": _code_version(),
        "runner_hash": _sha(Path(__file__).read_text(encoding="utf-8")),
        "song_hash": _sha(_json(snapshot)),
        "generation_prompt_hash": _sha(requests[0][1].system),
        "cleanup_prompt_hash": _sha(_prompt("paper_cleanup.txt")),
        "fidelity": {
            "original_exp4_instruction": variant == "paper-original",
            "human_annotations_in_generation": variant == "paper-annotated",
            "model_and_sampling": "current_project_configuration_not_original_paper_setup",
            "cleanup": "local_reconstruction_paper_does_not_publish_full_cleanup_prompt",
            "manual_verification": "pending",
        },
    }
    meta: dict[str, Any] = {"status": "preview" if dry_run else "running", "calls": []}
    if resume_dir is not None:
        previous_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        keys = ("song_hash", "variant", "model", "base_url", "protocol", "thinking_requested",
                "temperature_requested", "max_tokens", "generation_prompt_hash", "cleanup_prompt_hash")
        if any(previous_config.get(key) != config.get(key) for key in keys):
            raise ValueError("已有运行的输入、模型或生成设置不同，不能续跑")
        previous_meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        if previous_meta.get("status") == "completed":
            return run_dir
        meta["attempt_history"] = previous_meta.get("attempt_history", []) + [{
            "at": stamp, "previous_status": previous_meta.get("status"), "previous_error": previous_meta.get("error"),
            "resume_runner_hash": config["runner_hash"], "max_retries": config["max_retries"],
        }]
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        _write(run_dir / "song.json", snapshot)
        _write(run_dir / "config.json", config)
    _write(run_dir / "meta.json", meta)
    texts = {}
    try:
        for index, (section_id, request) in enumerate(requests, 1):
            section_dir = run_dir / "sections" / f"{index:03d}"
            _write(section_dir / "section.json", {"id": section_id, "phrase_ids": [p.id for p in song.phrases_of(section_id)]})
            if dry_run:
                _write(section_dir / "generation" / "request.json", request.to_json())
                continue
            print(f"[{variant}] {song.id} / {section_id} 生成", file=sys.stderr)
            generated = _call(client, request, section_dir / "generation", reuse=resume_dir is not None)
            meta["calls"].append({"section": section_id, "stage": "generation", "usage": generated["usage"], "reused": generated.get("reused", False)})
            cleaned = _call(client, cleanup_request(generated["response"]), section_dir / "cleanup", reuse=resume_dir is not None)
            meta["calls"].append({"section": section_id, "stage": "cleanup", "usage": cleaned["usage"], "reused": cleaned.get("reused", False)})
            phrases = song.phrases_of(section_id)
            lines, alignment_source = align_lines(generated["response"], cleaned["response"], [p.id for p in phrases])
            _write(section_dir / "alignment.json", {"source": alignment_source})
            _write(section_dir / "output.json", {"section": section_id, "phrases": [
                {"id": p.id, "zh": line} for p, line in zip(phrases, lines)
            ]})
            texts.update((p.id, line) for p, line in zip(phrases, lines))
            _write(run_dir / "meta.json", meta)
        if not dry_run:
            view, report = evaluate(song, texts, run_dir.name)
            _write(run_dir / "output.json", {"song_id": song.id, "phrases": report["phrases"]})
            _write(run_dir / "report.json", report)
            write_lyrics(song, view, run_dir)
            meta.update(status="completed", budget_pass_rate=report["budget_pass_rate"])
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        raise
    finally:
        _write(run_dir / "meta.json", meta)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="论文 Exp.4：原文组与人工标注组，共用当前模型")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "songs")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--song", action="append", help="歌曲 ID，可重复")
    targets.add_argument("--all", action="store_true", dest="all_songs")
    parser.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    parser.add_argument("--dry-run", action="store_true", help="只保存两组生成请求，不调用模型")
    parser.add_argument("--client", choices=("pending", "api"), default="pending", help="真实调用须显式选择 api")
    parser.add_argument("--model", default=None, help="默认使用 DUBBING_MODEL 或项目默认模型")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--protocol", choices=("anthropic", "openai"), default="anthropic", help="openai 使用 /chat/completions 和 Bearer 认证")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled", help="默认关闭思考；OpenAI 兼容接口传 enable_thinking=false")
    parser.add_argument("--resume", action="store_true", help="续跑各组最新记录；复用相同设置下已成功的调用，不重新抽样")
    parser.add_argument("--max-retries", type=int, default=3, help="每次接口调用的最大尝试次数，默认 3")
    parser.add_argument("--jobs", type=int, default=1, help="并发的歌曲 / baseline 任务数，默认 1；每个任务独立客户端")
    parser.add_argument("--temperature", type=float, default=None, help="默认使用当前项目配置；两组一致")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)
    if args.max_tokens < 1:
        parser.error("--max-tokens 必须为正整数")
    if args.jobs < 1:
        parser.error("--jobs 必须为正整数")
    if args.max_retries < 1:
        parser.error("--max-retries 必须为正整数")
    if args.resume and args.dry_run:
        parser.error("--resume 用于真实续跑，不能与 --dry-run 同用")
    if not args.dry_run and args.client != "api":
        print("需要 --client api 才会运行；可先使用 --dry-run 查看两组请求。", file=sys.stderr)
        return 2
    song_ids = list_songs(root=args.root) if args.all_songs else list(dict.fromkeys(args.song))
    if not song_ids:
        print("歌库为空。", file=sys.stderr)
        return 1
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    failures = []
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
        client = PaperClient(model=args.model, base_url=args.base_url, temperature=args.temperature,
                             max_tokens=args.max_tokens, protocol=args.protocol, thinking=args.thinking,
                             max_retries=args.max_retries)
        out = args.root / song_id / "baselines"
        candidates = sorted((out / variant).glob("run-*/config.json")) if args.resume else []
        resume_dir = candidates[-1].parent if candidates else None
        return run_one(song, variant, client, out=out, dry_run=args.dry_run, resume_dir=resume_dir)

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(pending):
            song_id, song, variant = pending[future]
            try:
                directory = future.result()
                print(f"{song_id} / {variant} -> {directory}", flush=True)
            except (OSError, ValueError, LLMError) as exc:
                failures.append(f"{song_id}/{variant}")
                print(f"{song_id} / {variant}: {exc}", file=sys.stderr)
    print(f"完成：{len(song_ids)} 首，{len(variants)} 组；失败 {len(failures)} 项。")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
