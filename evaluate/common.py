"""评测产物的序列化与复现标识。"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VARIANTS = ("paper-original", "paper-annotated", "workflow", "workflow-rag")
AVAILABLE_VARIANTS = (*VARIANTS, "one-shot", "one-shot-archive")
WRITER_VARIANTS = ("one-shot", "one-shot-archive")
VARIANT_LABELS = {
    "paper-original": "EACL Exp.4 原提示词",
    "paper-annotated": "原提示词 + 人工 budget / pattern",
    "workflow": "workflow（不加入 RAG）",
    "workflow-rag": "workflow-RAG（加入 RAG）",
    "one-shot": "writer 单次调用（无 perceiver、无 reviewer 返修）",
    "one-shot-archive": "归档 shared.md+b.md 的 writer 单次调用（无分析、无返修）",
}
DIMENSIONS = ("semantic", "emotion", "naturalness", "expressiveness")


def variants_of(snapshot):
    variants = tuple(snapshot.get("variants", VARIANTS))
    if not variants or len(set(variants)) != len(variants) or any(v not in AVAILABLE_VARIANTS for v in variants):
        raise ValueError("评测分组为空、重复或包含未知方法")
    return variants


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def average(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def label(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
