"""可审查、可续跑的匿名评分与成对顺序复评。"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import json
import random
import re
import shutil
from uuid import uuid4

from dubbing.llm import APIClient, LLMError, extract_json
from dubbing.models import Song

from .common import DIMENSIONS, VARIANTS, average, digest, file_hash, now, ratio, read, write, variants_of

PROMPTS = Path(__file__).parent / "prompts"


def build_tasks(snapshot, paired_songs, *, seed=0, limit=None):
    variants = variants_of(snapshot)
    rng = random.Random(seed)
    sections = []
    for item in snapshot["songs"]:
        if item["directory"] not in paired_songs:
            continue
        song = Song.from_json(item["song"])
        context = [{"section": s.id, "lyrics": [p.ja for p in song.phrases_of(s.id)]} for s in song.sections]
        for section in song.sections:
            phrases = song.translatable_of(section.id)
            if not phrases:
                continue
            source = {"source_context": context, "section": section.id,
                      "phrases": [{"id": p.id, "ja": p.ja, "budget": p.budget.to_json() if p.budget else None,
                                   "pattern": list(p.pattern) if p.pattern else None} for p in phrases]}
            sections.append((item, section.id, source, phrases))
    planned = len(sections)
    if limit is not None:
        sections = sections[:limit]
    tasks = []
    for item, section, source, phrases in sections:
        sid = item["directory"]
        order = list(variants)
        rng.shuffle(order)
        specs = [("score", dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", order)), None, None)]
        for a, b in combinations(variants, 2):
            comparison = digest([sid, section, a, b])[:20]
            specs.extend(("pair", {"A": x, "B": y}, comparison, orientation)
                         for orientation, (x, y) in enumerate(((a, b), (b, a))))
        for kind, mapping, comparison, orientation in specs:
            payload = {**source, "candidates": [
                {"label": label, "phrases": [{"id": p.id, "zh": item["candidates"][variant]["texts"][p.id]} for p in phrases]}
                for label, variant in mapping.items()
            ]}
            task = {"kind": kind, "song": sid, "section": section, "mapping": mapping,
                    "phrase_ids": [p.id for p in phrases], "comparison": comparison, "orientation": orientation,
                    "messages": {"system": (PROMPTS / f"{kind}.txt").read_text(encoding="utf-8").strip(),
                                 "user": json.dumps(payload, ensure_ascii=False, indent=2)}}
            task["id"] = digest(task)[:24]
            tasks.append(task)
    return tasks, {"planned_sections": planned, "selected_sections": len(sections), "planned_tasks": len(tasks)}


def _evidence(value, ids):
    if not isinstance(value, dict) or not isinstance(value.get("reason"), str) or not value["reason"].strip():
        raise ValueError("评审缺少非空简短理由")
    evidence = value.get("phrase_ids")
    if not isinstance(evidence, list) or any(not isinstance(pid, str) or pid not in ids for pid in evidence) or len(set(evidence)) != len(evidence):
        raise ValueError("评审引用了未知、重复或非法 phrase ID")
    return evidence


def validate_result(value, task):
    if not isinstance(value, dict):
        raise ValueError("评审输出必须为 JSON 对象")
    ids = set(task["phrase_ids"])
    if task["kind"] == "pair":
        if set(value) != {"winner", "reason", "phrase_ids"} or value["winner"] not in ("A", "B", "tie", None):
            raise ValueError("成对评审 winner 只能是 A、B、tie 或 null")
        evidence = _evidence(value, ids)
        if value["winner"] is not None and not evidence:
            raise ValueError("有效偏好必须引用乐句依据")
    else:
        rows = value.get("candidates")
        if set(value) != {"candidates"} or not isinstance(rows, list) or len(rows) != len(task["mapping"]):
            raise ValueError("评分必须恰好覆盖所有匿名候选")
        labels = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("label"), str) or not isinstance(row.get("scores"), dict):
                raise ValueError("评分候选格式错误")
            labels.append(row["label"])
            if set(row["scores"]) != set(DIMENSIONS):
                raise ValueError("评分必须覆盖且仅覆盖四个文本维度")
            for item in row["scores"].values():
                evidence = _evidence(item, ids)
                score = item.get("score")
                if set(item) != {"score", "reason", "phrase_ids"} or score is not None and (type(score) is not int or not 1 <= score <= 5):
                    raise ValueError("评分只能为 1–5 的整数或 null")
                if score is not None and not evidence:
                    raise ValueError("有效评分必须引用乐句依据")
        if len(set(labels)) != len(labels) or set(labels) != set(task["mapping"]):
            raise ValueError("评分标签重复、缺失或未知")
    return value


def client_config(client):
    return {"model": getattr(client, "model", client.name), "base_url": getattr(client, "base_url", None),
            "protocol": getattr(client, "protocol", None), "thinking": getattr(client, "thinking", None),
            "temperature": client.temperature_for("evaluation") if isinstance(client, APIClient) else None,
            "max_tokens": getattr(client, "max_tokens", None)}


def _redact(value, client):
    if isinstance(value, str):
        key = getattr(client, "api_key", None)
        if key:
            value = value.replace(key, "[REDACTED]")
        return re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[REDACTED]", value)
    if isinstance(value, dict):
        return {k: _redact(v, client) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, client) for v in value]
    return value


def request_for(task, client):
    request = {"messages": task["messages"], "local": {k: v for k, v in task.items() if k != "messages"}}
    if isinstance(client, APIClient):
        request["transport"] = {"url": client.url, "body": client._body(**task["messages"], agent="evaluation")}
    return request


def execute_task(task, client, directory, *, dry_run=False):
    directory = Path(directory)
    request = request_for(task, client)
    if (directory / "request.json").exists() and read(directory / "request.json") != request:
        raise ValueError("已有评审请求与当前设置不同，拒绝复用")
    if (directory / "meta.json").exists() and read(directory / "meta.json")["status"] == "completed":
        raw = (directory / "response.txt").read_text(encoding="utf-8")
        previous = read(directory / "meta.json")
        if digest(raw) != previous.get("response_hash"):
            raise ValueError("已有评审响应被修改，不能复用")
        value = validate_result(extract_json(raw), task)
        if read(directory / "result.json") != value:
            raise ValueError("已有评审解析结果被修改，不能复用")
        return {"status": "completed", "result": value, "reused": True}
    if dry_run:
        write(directory / "request.json", request)
        return {"status": "preview"}
    if (directory / "meta.json").exists():
        archive = directory / "attempts" / (now().replace(":", "") + "-" + uuid4().hex[:6])
        archive.mkdir(parents=True)
        for path in directory.iterdir():
            if path.is_file():
                shutil.copy2(path, archive / path.name)
    write(directory / "request.json", request)
    meta = {"status": "running", "started_at": now()}
    write(directory / "meta.json", meta)
    try:
        response = client.complete(agent="evaluation", **task["messages"])
        response = _redact(response, client)
        (directory / "response.txt").write_text(response, encoding="utf-8")
        meta["response_hash"] = digest(response)
        value = validate_result(extract_json(response), task)
        write(directory / "result.json", value)
        meta["status"] = "completed"
        return {"status": "completed", "result": value, "reused": False}
    except (LLMError, ValueError, OSError) as exc:
        meta.update(status="failed", error=_redact(str(exc), client))
        return {"status": "failed", "error": meta["error"]}
    finally:
        meta.update(finished_at=now(), usage=getattr(client, "last_usage", None),
                    thinking_observation=getattr(client, "last_thinking_observation", None))
        if getattr(client, "last_response", None) is not None:
            write(directory / "response.json", _redact(client.last_response, client))
        write(directory / "meta.json", _redact(meta, client))


def summarize(tasks, outcomes, *, variants=VARIANTS):
    scores, pairs = [], defaultdict(list)
    for task, outcome in zip(tasks, outcomes, strict=True):
        if task["kind"] == "score":
            if outcome["status"] == "completed":
                for row in outcome["result"]["candidates"]:
                    scores.append({"song": task["song"], "section": task["section"], "variant": task["mapping"][row["label"]], "scores": row["scores"]})
        else:
            pairs[task["comparison"]].append((task, outcome))
    absolute = {}
    score_tasks = [t for t in tasks if t["kind"] == "score"]
    for variant in variants:
        absolute[variant] = {}
        for dimension in DIMENSIONS:
            by_song = defaultdict(list)
            for row in scores:
                if row["variant"] == variant and row["scores"][dimension]["score"] is not None:
                    by_song[row["song"]].append(row["scores"][dimension]["score"])
            sections = sum(len(v) for v in by_song.values())
            absolute[variant][dimension] = {"macro_score": average([average(v) for v in by_song.values()]),
                                             "valid_songs": len(by_song), "valid_sections": sections,
                                             "planned_sections": len(score_tasks), "coverage": ratio(sections, len(score_tasks))}
    absolute_paired = {v: {} for v in variants}
    for dimension in DIMENSIONS:
        keys = {
            v: {(r["song"], r["section"]) for r in scores if r["variant"] == v and r["scores"][dimension]["score"] is not None}
            for v in variants
        }
        common = set.intersection(*keys.values())
        for variant in variants:
            by_song = defaultdict(list)
            for row in scores:
                if row["variant"] == variant and (row["song"], row["section"]) in common:
                    by_song[row["song"]].append(row["scores"][dimension]["score"])
            absolute_paired[variant][dimension] = {"macro_score": average([average(v) for v in by_song.values()]),
                                                   "valid_songs": len(by_song), "valid_sections": len(common),
                                                   "planned_sections": len(score_tasks), "coverage": ratio(len(common), len(score_tasks))}
    comparisons = []
    for entries in pairs.values():
        task = entries[0][0]
        a, b = sorted(task["mapping"].values(), key=variants.index)
        row = {"song": task["song"], "section": task["section"], "a": a, "b": b}
        if len(entries) != 2 or any(o["status"] != "completed" for _, o in entries):
            row["status"] = "missing"
        elif any(o["result"]["winner"] is None for _, o in entries):
            row["status"] = "abstained"
        else:
            winners = ["tie" if o["result"]["winner"] == "tie" else t["mapping"][o["result"]["winner"]] for t, o in entries]
            row.update(status="stable" if winners[0] == winners[1] else "unstable", winners=winners)
            if row["status"] == "stable":
                row["winner"] = winners[0]
        comparisons.append(row)
    preferences = []
    for a, b in combinations(variants, 2):
        rows = [r for r in comparisons if r["a"] == a and r["b"] == b]
        stable = [r for r in rows if r["status"] == "stable"]
        by_song = defaultdict(list)
        for row in stable:
            by_song[row["song"]].append(.5 if row["winner"] == "tie" else int(row["winner"] == a))
        preferences.append({"a": a, "b": b, "planned_pairs": len(rows), "stable_pairs": len(stable),
                            "unstable_pairs": sum(r["status"] == "unstable" for r in rows),
                            "abstained_pairs": sum(r["status"] == "abstained" for r in rows),
                            "missing_pairs": sum(r["status"] == "missing" for r in rows),
                            "stable_coverage": ratio(len(stable), len(rows)),
                            "win_rate_a": average([average(v) for v in by_song.values()]),
                            "micro_win_rate_a": average([.5 if r["winner"] == "tie" else int(r["winner"] == a) for r in stable])})
    focus = next((p for p in preferences if p["a"] == "workflow" and p["b"] == "workflow-rag"), None)
    rag_preference = None
    if focus is not None:
        rag_preference = {"a": "workflow-rag", "b": "workflow",
                          "win_rate_a": 1 - focus["win_rate_a"] if focus["win_rate_a"] is not None else None,
                          "micro_win_rate_a": 1 - focus["micro_win_rate_a"] if focus["micro_win_rate_a"] is not None else None,
                          **{k: focus[k] for k in ("planned_pairs", "stable_pairs", "unstable_pairs", "abstained_pairs", "missing_pairs", "stable_coverage")}}
    return {"absolute": absolute, "absolute_paired": absolute_paired, "scores": scores, "preferences": preferences,
            "rag_vs_workflow": rag_preference, "comparisons": comparisons,
            "completed_tasks": sum(o["status"] == "completed" for o in outcomes),
            "failed_tasks": sum(o["status"] == "failed" for o in outcomes),
            "preview_tasks": sum(o["status"] == "preview" for o in outcomes),
            "task_count": len(tasks), "human_evaluation": False}


def reuse_tasks(tasks, client, out, source, config):
    """只复制请求、模型与提示词完全一致的完成任务，不混用旧的汇总分。"""
    source = Path(source).resolve()
    previous = read(source / "config.json")
    for key in ("seed", "limit", "prompts", "client"):
        if previous.get(key) != config[key]:
            raise ValueError(f"复用评审的 {key} 设置不同，不能混合比较")
    reused = []
    for task in tasks:
        old = source / "tasks" / task["id"]
        destination = out / "tasks" / task["id"]
        if destination.exists() or not (old / "meta.json").is_file():
            continue
        if read(old / "meta.json").get("status") != "completed":
            continue
        if read(old / "request.json") != request_for(task, client):
            continue
        # 在复制前验证正文哈希、解析结果及证据；此路径不会调用模型。
        execute_task(task, client, old, dry_run=True)
        destination.mkdir(parents=True)
        for name in ("request.json", "response.txt", "response.json", "result.json", "meta.json"):
            path = old / name
            if path.is_file():
                shutil.copy2(path, destination / name)
        reused.append(task["id"])
    existing = read(out / "reused-from.json") if (out / "reused-from.json").is_file() else {}
    prior_ids = existing.get("task_ids", []) if existing.get("directory") == str(source) else []
    write(out / "reused-from.json", {"directory": str(source), "config_hash": digest(previous),
                                    "task_ids": sorted(set(prior_ids + reused)),
                                    "note": "仅复用完整请求与配置相同的已完成任务，未复用汇总评分。"})


def run_judges(snapshot, rules, client, out, *, seed=0, limit=None, dry_run=False, jobs=1, client_factory=None, reuse_from=None):
    if jobs < 1 or jobs > 1 and client_factory is None:
        raise ValueError("并发评审需要正数 jobs 及独立客户端工厂")
    out = Path(out)
    config = {"input_hash": snapshot["input_hash"], "seed": seed, "limit": limit,
              "prompts": {name: file_hash(PROMPTS / f"{name}.txt") for name in ("score", "pair")},
              "client": client_config(client)}
    if (out / "config.json").exists() and read(out / "config.json") != config:
        raise ValueError("评审模型、提示词、输入或选择设置已变化，请使用新的评测目录")
    write(out / "config.json", config)
    tasks, selection = build_tasks(snapshot, rules["paired_songs"], seed=seed, limit=limit)
    if reuse_from is not None:
        reuse_tasks(tasks, client, out, reuse_from, config)
    outcomes = [{"status": "pending"} for _ in tasks]

    def checkpoint():
        pending = sum(o["status"] == "pending" for o in outcomes)
        status = "running" if pending else "preview" if dry_run else "completed" if all(o["status"] == "completed" for o in outcomes) else "partial"
        result = {"status": status, "selection": selection, "execution": {"jobs": jobs},
                  "pending_tasks": pending, **summarize(tasks, outcomes, variants=variants_of(snapshot))}
        write(out / "summary.json", result)
        return result

    def execute(index):
        task = tasks[index]
        worker = client_factory() if jobs > 1 else client
        return execute_task(task, worker, out / "tasks" / task["id"], dry_run=dry_run)

    def record(index, result, completed):
        outcomes[index] = result
        task = tasks[index]
        print(f"评审 {completed}/{len(tasks)} {task['song']}/{task['section']} {task['kind']}: {result['status']}", flush=True)
        if completed == 1 or completed % 10 == 0:
            checkpoint()

    checkpoint()
    if jobs == 1:
        for index in range(len(tasks)):
            record(index, execute(index), index + 1)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(execute, index): index for index in range(len(tasks))}
            for completed, future in enumerate(as_completed(futures), 1):
                record(futures[future], future.result(), completed)
    return checkpoint()
