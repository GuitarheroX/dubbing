"""从固定运行中读取译词，保留缺失与对齐错误；不修改生成结果。"""

from collections import Counter
from pathlib import Path

from dubbing.models import Song
from dubbing.song import list_songs, load_song
import yaml

from dubbing.store import RunStore

from .common import REPO_ROOT, VARIANTS, WRITER_VARIANTS, digest, file_hash, label, read, variants_of


def projection(song):
    return {
        "sections": song.section_ids,
        "phrases": [{k: p.to_json().get(k) for k in ("id", "section", "ja", "budget", "pattern", "keep_source")} for p in song.phrases],
    }


class Reader:
    def __init__(self):
        self.files = {}

    def record(self, path):
        path = Path(path)
        if path.is_file():
            self.files[label(path)] = file_hash(path)

    def json(self, path):
        self.record(path)
        return read(path)


def aligned(rows, expected, issues, where):
    """重复 ID 的全部译文作废；未知 ID 不计覆盖，问题明确留档。"""
    if not isinstance(rows, list):
        issues.append(f"{where}: phrases 不是数组")
        return {}
    ids = [r.get("id") for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)]
    counts = Counter(ids)
    texts = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str):
            issues.append(f"{where}: 乐句缺少字符串 ID")
            continue
        pid = r["id"]
        if pid not in expected or counts[pid] != 1:
            issues.append(f"{where}: 未知或重复 ID {pid}")
            continue
        text = r.get("zh")
        if not isinstance(text, str) or not text.strip():
            issues.append(f"{where}: {pid} 译词为空或不是字符串")
        elif "〈未译〉" in text or text.startswith("【未完成"):
            issues.append(f"{where}: {pid} 是导出占位符")
        else:
            texts[pid] = text.strip()
    missing = sorted(set(expected) - texts.keys())
    if missing:
        issues.append(f"{where}: 缺少有效译词 {', '.join(missing)}")
    return texts


def _metadata(meta):
    return {k: meta[k] for k in ("agent", "model", "temperature", "temperature_requested", "protocol", "thinking_requested", "prompt_hash", "round", "finished_at") if k in meta}


def _validate_source(rows, phrases, where):
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("id"), str) for r in rows):
        raise ValueError(f"{where}: 输入格式错误")
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {p.id for p in phrases}:
        raise ValueError(f"{where}: 输入乐句 ID 与当前数据不同或重复")
    for p in phrases:
        row = by_id[p.id]
        for k in ("ja", "budget", "pattern", "keep_source"):
            default = False if k == "keep_source" else None
            if row.get(k, default) != p.to_json().get(k, default):
                raise ValueError(f"{where}: 主流程/标注输入版本不同 ({p.id})")


def _legacy_ours(root, sid, song, reader):
    state_path = root / sid / "state.yaml"
    reader.record(state_path)
    state = yaml.safe_load(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    state = state or {}
    sections = state.get("sections", {})
    analysis = state.get("song_level", {}).get("analysis") or state.get("analysis")
    runs_dir = root / sid / "runs"
    texts, issues, sources, models = {}, [], {}, []
    if analysis:
        for name in ("output.json", "meta.json"):
            reader.record(runs_dir / analysis / name)
    for section in song.sections:
        ss = sections.get(section.id, {})
        expected = {p.id for p in song.translatable_of(section.id)}
        if not expected:
            continue
        draft = ss.get("draft")
        if not draft:
            issues.append(f"{section.id}: 无当前 draft")
            continue
        directory = runs_dir / draft
        if not directory.resolve().is_relative_to(runs_dir.resolve()):
            raise ValueError(f"{sid}/{section.id}: draft 路径超出 runs 目录")
        if not (directory / "output.json").is_file():
            issues.append(f"{section.id}: 无当前 draft 输出")
            continue
        output = reader.json(directory / "output.json")
        if output.get("section") != section.id:
            issues.append(f"{section.id}: draft 段落不符")
            continue
        request_path = directory / "request.json"
        if not request_path.is_file():
            issues.append(f"{section.id}: 缺少输入快照，无法核实数据版本")
            continue
        data = reader.json(request_path).get("data", {})
        _validate_source(data.get("phrases"), song.phrases, f"{sid}/{section.id}")
        texts.update(aligned(output.get("phrases"), expected, issues, section.id))
        sources[section.id] = {"draft": draft, "review": ss.get("review")}
        if (directory / "meta.json").is_file():
            models.append(_metadata(reader.json(directory / "meta.json")))
    needed = {p.id for p in song.phrases if p.translatable}
    texts.update({p.id: p.ja for p in song.phrases if p.keep_source})
    completed = needed <= texts.keys() and not issues
    return {"texts": texts, "issues": issues, "completed": completed,
            "status": "completed" if completed else "partial",
            "source": {"layout": "legacy", "sections": sources, "analysis": analysis}, "models": models}


def _ours(root, sid, song, reader, pipeline):
    base = root / sid / ("workflow_rag" if pipeline == "rag" else "workflow")
    store = RunStore(base)
    reader.record(store.latest_path)
    run_id = store.latest_run()
    if run_id is None:
        if pipeline == "workflow":
            return _legacy_ours(root, sid, song, reader)
        return {"texts": {}, "issues": ["无 RAG 运行记录"], "completed": False,
                "status": "missing", "source": {"pipeline": pipeline}, "models": []}
    directory = store.run_dir(run_id).resolve()
    if not directory.is_relative_to(base.resolve()) or directory == base.resolve():
        raise ValueError(f"{sid}: run 路径超出管线目录")
    latest = store.read_latest()
    if latest.get("run") != run_id:
        latest = {}
    texts, issues, models, record_agents = {}, [], [], {}
    analysis_agent = "perceiver" if (directory / "perceiver").is_dir() else "A"
    for name in ("request.json", "output.json", "meta.json"):
        reader.record(directory / analysis_agent / name)
    agent = "mimic_write" if pipeline == "rag" else "writer"
    for index, section in enumerate(song.sections, 1):
        expected = {p.id for p in song.translatable_of(section.id)}
        if not expected:
            continue
        record = directory / "sections" / f"{index:03d}" / agent
        if pipeline == "workflow" and not (record / "output.json").exists():
            old_record = record.parent / "B"
            if (old_record / "output.json").is_file():
                record = old_record
        record_agents[section.id] = record.name
        if not (record / "output.json").is_file() or not (record / "request.json").is_file():
            issues.append(f"{section.id}: 无 {agent} 输入或输出")
            continue
        data = reader.json(record / "request.json").get("data", {})
        _validate_source(data.get("phrases"), song.phrases, f"{sid}/{section.id}")
        meta = reader.json(record / "meta.json") if (record / "meta.json").is_file() else {}
        models.append(_metadata(meta))
        if meta.get("abandoned"):
            issues.append(f"{section.id}: {meta['abandoned']}")
            continue
        output = reader.json(record / "output.json")
        if output.get("section") != section.id:
            issues.append(f"{section.id}: 段落不符")
            continue
        texts.update(aligned(output.get("phrases"), expected, issues, section.id))
        review_dir = directory / "sections" / f"{index:03d}" / "reviewer"
        if not review_dir.is_dir():
            review_dir = review_dir.parent / "C"
        for name in ("request.json", "output.json", "meta.json"):
            reader.record(review_dir / name)
    texts.update({p.id: p.ja for p in song.phrases if p.keep_source})
    completed = {p.id for p in song.phrases if p.translatable} <= texts.keys() and not issues
    return {"texts": texts, "issues": issues, "completed": completed,
            "status": latest.get("status", "completed" if completed else "partial"),
            "source": {"layout": "unified", "pipeline": pipeline, "run": label(directory), "draft_agent": agent,
                       "record_agents": record_agents}, "models": models}


def _baseline(root, sid, song, variant, selected, reader):
    parent = (root / sid / "baselines" / variant).resolve()
    if selected is None:
        pointer = parent / "latest.json"
        if not pointer.is_file():
            return {"texts": {}, "issues": ["无 baseline 记录"], "completed": False, "status": "missing", "source": None, "models": []}
        selected = parent / reader.json(pointer)["run"]
    directory = Path(selected)
    directory = (directory if directory.is_absolute() else REPO_ROOT / directory).resolve()
    if not directory.is_relative_to(parent) or directory == parent:
        raise ValueError(f"{sid}/{variant}: run 路径必须位于对应 baseline 目录")
    if not (directory / "song.json").is_file() and not (directory / "config.json").is_file():
        return _unified_baseline(directory, sid, song, variant, parent, reader)
    frozen = Song.from_json(reader.json(directory / "song.json"))
    if projection(frozen) != projection(song):
        raise ValueError(f"{sid}/{variant}: baseline 与当前输入版本不同")
    config = reader.json(directory / "config.json")
    if config.get("variant") != variant:
        raise ValueError(f"{sid}/{variant}: 配置中的组别不一致")
    meta = reader.json(directory / "meta.json")
    texts, issues, seen_sections = {}, [], set()
    for section_dir in sorted((directory / "sections").glob("*")):
        if not (section_dir / "output.json").is_file():
            continue
        output = reader.json(section_dir / "output.json")
        section_id = output.get("section")
        if section_id not in song.section_ids or section_id in seen_sections:
            raise ValueError(f"{sid}/{variant}: 未知或重复段落 {section_id}")
        seen_sections.add(section_id)
        expected = {p.id for p in song.phrases_of(section_id)}
        texts.update(aligned(output.get("phrases"), expected, issues, section_id))
    # 完成运行的全曲文件仅作一致性校验，不用导出文本填补未对齐段落。
    if meta.get("status") == "completed":
        if not (directory / "output.json").is_file():
            issues.append("完成运行缺少全曲 output.json")
        else:
            full = reader.json(directory / "output.json").get("phrases", [])
            raw_rows = [{"id": p.get("id"), "zh": p.get("model_text")} for p in full if isinstance(p, dict)] if isinstance(full, list) else None
            final_texts = aligned(raw_rows, {p.id for p in song.phrases}, issues, "全曲输出")
            if final_texts != texts:
                raise ValueError(f"{sid}/{variant}: 全曲输出与逐段原始译词不一致")
    needed = {p.id for p in song.phrases if p.translatable}
    completed = meta.get("status") == "completed" and needed <= texts.keys() and not issues
    return {"texts": texts, "issues": issues, "completed": completed,
            "status": meta.get("status", "unknown"), "error": meta.get("error"),
            "source": label(directory), "models": [_metadata(config)]}


def _unified_baseline(directory, sid, song, variant, parent, reader):
    texts, issues, models = {}, [], []
    reader.record(parent / "latest.json")
    latest = RunStore(parent).read_latest()
    if latest.get("run") != directory.name:
        latest = {}
    for index, section in enumerate(song.sections, 1):
        record = directory / "sections" / f"{index:03d}" / "generation"
        if variant in WRITER_VARIANTS:
            if not song.translatable_of(section.id):
                continue
            record = record.parent / "writer"
            if not (record / "output.json").is_file() and (record.parent / "B" / "output.json").is_file():
                record = record.parent / "B"
        if not (record / "output.json").is_file() or not (record / "request.json").is_file():
            issues.append(f"{section.id}: 无生成输入或输出")
            continue
        data = reader.json(record / "request.json").get("data", {})
        if variant in WRITER_VARIANTS:
            _validate_source(data.get("phrases"), song.phrases, f"{sid}/{variant}/{section.id}")
            if data.get("focus") != section.id:
                raise ValueError(f"{sid}/{variant}/{section.id}: one-shot 输入 focus 与段落不同")
        elif variant == "paper-original":
            if data.get("lyrics") != [p.ja for p in song.phrases_of(section.id)]:
                raise ValueError(f"{sid}/{variant}/{section.id}: baseline 原文输入版本不同")
        else:
            _validate_source(data.get("phrases"), song.phrases_of(section.id), f"{sid}/{variant}/{section.id}")
        meta = reader.json(record / "meta.json") if (record / "meta.json").is_file() else {}
        models.append(_metadata(meta))
        cleanup = record.parent / "cleanup"
        for name in ("request.json", "output.json", "meta.json"):
            reader.record(cleanup / name)
        if meta.get("abandoned"):
            issues.append(f"{section.id}: {meta['abandoned']}")
            continue
        output = reader.json(record / "output.json")
        if output.get("section") != section.id or variant not in WRITER_VARIANTS and output.get("source") not in ("cleanup.lines", "generation.phrases.zh"):
            issues.append(f"{section.id}: 输出尚未对齐或段落不符")
            continue
        expected = song.translatable_of(section.id) if variant in WRITER_VARIANTS else song.phrases_of(section.id)
        texts.update(aligned(output.get("phrases"), {p.id for p in expected}, issues, section.id))
    if variant in WRITER_VARIANTS:
        texts.update({p.id: p.ja for p in song.phrases if p.keep_source})
    completed = {p.id for p in song.phrases if p.translatable} <= texts.keys() and not issues
    return {"texts": texts, "issues": issues, "completed": completed,
            "status": latest.get("status", "completed" if completed else "partial"),
            "source": {"layout": "unified", "run": label(directory),
                       "annotation_validation": "source_only_original_did_not_receive_annotations" if variant == "paper-original" else "annotated_requests"},
            "models": models}


def collect(root, song_ids=None, batch=None, *, workflow_root=None, rag_root=None, baseline_root=None, variants=VARIANTS):
    root = Path(root).resolve()
    variants = variants_of({"variants": variants})
    roots = {
        "paper-original": Path(baseline_root or root).resolve(),
        "paper-annotated": Path(baseline_root or root).resolve(),
        "workflow": Path(workflow_root or root).resolve(),
        "workflow-rag": Path(rag_root or root).resolve(),
        "one-shot": Path(baseline_root or root).resolve(),
        "one-shot-archive": Path(baseline_root or root).resolve(),
    }
    roots = {v: roots[v] for v in variants}
    reader = Reader()
    selected = None
    if batch:
        selected = {}
        for row in reader.json(batch)["results"]:
            key = (row["song"], row["variant"])
            if key in selected:
                raise ValueError(f"批次重复记录: {key}")
            selected[key] = row["directory"]
    song_ids = list(dict.fromkeys(song_ids or list_songs(root=root)))
    if not song_ids:
        raise ValueError("歌库为空")
    songs = []
    for sid in song_ids:
        song = load_song(sid, root=root)
        reader.record(root / sid / "song.yaml")
        for source_root in set(roots.values()):
            source_song = load_song(sid, root=source_root)
            reader.record(source_root / sid / "song.yaml")
            if projection(source_song) != projection(song):
                raise ValueError(f"{sid}: 四组源歌词、分句或人工标注版本不同，不能混合比较")
        candidates = {}
        for variant in variants:
            if variant in ("workflow", "workflow-rag"):
                candidates[variant] = _ours(roots[variant], sid, song, reader, "rag" if variant == "workflow-rag" else "workflow")
                continue
            if selected is not None and (sid, variant) not in selected:
                raise ValueError(f"批次缺少 {sid}/{variant}")
            candidates[variant] = _baseline(roots[variant], sid, song, variant, selected.get((sid, variant)) if selected is not None else None, reader)
        songs.append({"directory": sid, "song": song.to_json(), "candidates": candidates})
    for path, fingerprint in reader.files.items():
        p = Path(path)
        p = p if p.is_absolute() else REPO_ROOT / p
        if not p.is_file() or file_hash(p) != fingerprint:
            raise ValueError(f"读取评测输入期间文件发生变化: {path}")
    return {"schema_version": 2, "variants": list(variants), "songs": songs,
            "selection": {"root": str(root), "roots": {k: str(v) for k, v in roots.items()}, "batch": label(batch) if batch else None},
            "input_files": reader.files, "input_hash": digest(songs)}
