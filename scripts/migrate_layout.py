#!/usr/bin/env python3
"""一次性迁移：把旧布局整理成 ``<song>/{song.yaml, baselines/, workflow/}``。

做三件事（默认只预览，加 ``--apply`` 才落盘）：

1. **workflow**：旧 ``runs/<ts>_<agent>_<hash>/`` + ``state.yaml`` 合成一个 run 目录
   ``workflow/run-<ts>Z-<hash>/``，A 放 run 根，B/C 放 ``sections/NNN/``；每个 agent
   只留 ``request.json`` / ``output.json`` / ``meta.json``（原始正文进 meta 的
   ``raw_response``）。随后删掉 ``runs/`` ``archive/`` ``state.yaml`` ``preview/``
   和歌曲根目录的 ``lyrics.zh.txt`` / ``lyrics.md``。
2. **paper baseline**：``baselines/paper-*/run-*/`` 去掉 ``config.json`` ``song.json``
   ``meta.json`` ``output.json`` ``report.json``、逐段的 ``section.json`` /
   ``alignment.json`` / ``output.json`` 和所有 ``response.txt`` / ``response.json``；
   段落结论搬进 ``generation/output.json``，清理结果写进 ``cleanup/output.json``。
3. **清理**：删除旧的 ``baseline.txt``（10 首）和 ``baseline/prompt.txt``——那是旧
   prompt、旧粒度的产物，与新的 one-shot 不可比，需要时从 git 历史取回。

用法：

    python scripts/migrate_layout.py --dry-run
    python scripts/migrate_layout.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dubbing.export import write_lyrics  # noqa: E402
from dubbing.models import Song  # noqa: E402
from dubbing.song import list_songs, load_song  # noqa: E402
from dubbing.store import RunStore, run_base, section_index  # noqa: E402
from dubbing.view import build_view  # noqa: E402

AGENTS = ("a", "b", "c")
PAPER_VARIANTS = ("paper-original", "paper-annotated")


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def say(apply: bool, message: str) -> None:
    print(("[改] " if apply else "[预] ") + message)


# --------------------------------------------------------------------------
# 1. workflow
# --------------------------------------------------------------------------


def _agent_of(run_id: str) -> str:
    parts = run_id.split("_")
    return parts[1] if len(parts) > 1 else ""


def _section_of(run_dir: Path) -> str | None:
    payload = read_json(run_dir / "output.json")
    if isinstance(payload, dict):
        section = payload.get("section")
        if isinstance(section, str):
            return section
    return None


def _draft_of(run_dir: Path) -> str | None:
    payload = read_json(run_dir / "output.json")
    if isinstance(payload, dict):
        draft = payload.get("draft_run")
        if isinstance(draft, str):
            return draft
    return None


def collect_pointers(runs_dir: Path, song: Song) -> dict[str, Any]:
    """优先用 ``state.yaml``；缺的或不存在的指针用 runs 现场重建补齐。"""
    run_ids = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    by_agent: dict[str, list[str]] = {a: [] for a in AGENTS}
    for rid in run_ids:
        agent = _agent_of(rid)
        if agent in by_agent:
            by_agent[agent].append(rid)

    # 重建：每个 section 取时间上最后一个 B；C 优先取评审该 B 的那个
    best_b: dict[str, str] = {}
    for rid in by_agent["b"]:
        sid = _section_of(runs_dir / rid)
        if sid:
            best_b[sid] = rid
    best_c: dict[str, str] = {}
    for rid in by_agent["c"]:
        sid = _section_of(runs_dir / rid)
        if not sid:
            continue
        if _draft_of(runs_dir / rid) == best_b.get(sid):
            best_c[sid] = rid
    for rid in by_agent["c"]:
        sid = _section_of(runs_dir / rid)
        if sid and sid not in best_c:
            best_c[sid] = rid

    state = {}
    state_path = runs_dir.parent / "state.yaml"
    if state_path.is_file():
        try:
            import yaml

            state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 - 迁移脚本，坏了就退回重建
            state = {}

    analysis = (state.get("song_level") or {}).get("analysis")
    if not (isinstance(analysis, str) and (runs_dir / analysis).is_dir()):
        analysis = by_agent["a"][-1] if by_agent["a"] else None

    drafts: dict[str, str | None] = {}
    reviews: dict[str, str | None] = {}
    entries = state.get("sections") or {}
    for sid in song.section_ids:
        entry = entries.get(sid) or {}
        draft = entry.get("draft")
        review = entry.get("review")
        drafts[sid] = draft if isinstance(draft, str) and (runs_dir / draft).is_dir() else best_b.get(sid)
        reviews[sid] = review if isinstance(review, str) and (runs_dir / review).is_dir() else best_c.get(sid)
    return {"analysis": analysis, "drafts": drafts, "reviews": reviews}


def _copy_record(runs_dir: Path, store: RunStore, run_id: str, src: str, agent: str, index: int | None) -> None:
    src_dir = runs_dir / src
    dst = store.record_dir(run_id, agent.upper(), index)
    dst.mkdir(parents=True, exist_ok=True)
    request = read_json(src_dir / "request.json")
    if request is not None:
        write_json(dst / "request.json", request)
    output = read_json(src_dir / "output.json")
    if output is not None:
        write_json(dst / "output.json", output)
    meta = read_json(src_dir / "meta.json")
    meta = dict(meta) if isinstance(meta, dict) else {}
    meta.setdefault("agent", agent.upper())
    raw = read_text(src_dir / "response.txt")
    if raw is not None:
        meta["raw_response"] = raw
    meta["migrated_from"] = src
    write_json(dst / "meta.json", meta)


def migrate_workflow(song: Song, song_dir: Path, root: Path, *, apply: bool) -> bool:
    runs_dir = song_dir / "runs"
    if not runs_dir.is_dir():
        return False
    pointers = collect_pointers(runs_dir, song)
    used = [pointers["analysis"], *pointers["drafts"].values(), *pointers["reviews"].values()]
    used = [rid for rid in used if rid]
    if not used:
        say(apply, f"{song.id}: runs/ 里没有可用的产物，跳过 workflow 迁移")
        return False

    stamp = max(rid.split("_")[0] for rid in used)
    # 路径一律按目录名（= list_songs 的 song_id），别用 song.yaml 里的 id，
    # 两者不一定相同（例如 bara/ 的 song.id 是 bara_wa_utsukushiku_chiru）。
    dirname = song_dir.name
    digest = hashlib.sha256(dirname.encode("utf-8")).hexdigest()[:8]
    run_id = f"run-{stamp}Z-{digest}"
    store = RunStore(run_base(root, dirname, "workflow"))
    say(apply, f"{song.id}: workflow -> {run_id}（A={pointers['analysis']}）")
    if not apply:
        for sid in song.section_ids:
            if pointers["drafts"].get(sid) or pointers["reviews"].get(sid):
                print(
                    f"      sections/{section_index(song, sid):03d}  "
                    f"B={pointers['drafts'].get(sid)}  C={pointers['reviews'].get(sid)}"
                )
        return True

    store.run_dir(run_id).mkdir(parents=True, exist_ok=True)
    if pointers["analysis"]:
        _copy_record(runs_dir, store, run_id, pointers["analysis"], "a", None)
    for sid in song.section_ids:
        index = section_index(song, sid)
        if pointers["drafts"].get(sid):
            _copy_record(runs_dir, store, run_id, pointers["drafts"][sid], "b", index)
        if pointers["reviews"].get(sid):
            _copy_record(runs_dir, store, run_id, pointers["reviews"][sid], "c", index)

    view = build_view(song, store, run_id)
    write_lyrics(song, view, store.run_dir(run_id))
    verdicts = {r.verdict for r in view.rows}
    model = None
    meta = store.read_meta(run_id, "A") if store.has_output(run_id, "A") else {}
    if isinstance(meta, dict):
        model = meta.get("model")
    store.set_latest(
        run_id,
        song=dirname,
        status="completed" if verdicts <= {"pass", "skip"} else "incomplete",
        model=model,
        migrated=True,
    )
    return True


def cleanup_workflow_leftovers(song_dir: Path, *, apply: bool) -> None:
    targets = [
        song_dir / "runs",
        song_dir / "archive",
        song_dir / "preview",
        song_dir / "state.yaml",
        song_dir / "view.json",
        song_dir / "lyrics.zh.txt",
        song_dir / "lyrics.md",
        song_dir / "baseline.txt",
    ]
    for path in targets:
        if not path.exists():
            continue
        say(apply, f"{song_dir.name}: 删除 {path.name}")
        if apply:
            shutil.rmtree(path) if path.is_dir() else path.unlink()


# --------------------------------------------------------------------------
# 2. paper baseline 瘦身
# --------------------------------------------------------------------------


CONFIG_KEEP = {
    "model": None,
    "base_url": None,
    "protocol": None,
    "thinking_requested": None,
    "temperature_requested": None,
    "max_tokens": None,
    "paper": None,
    "experiment": None,
    "fidelity": None,
}


def _fold_config(config: dict[str, Any], run_name: str, variant: str, agent: str) -> dict[str, Any]:
    meta = {key: config.get(key) for key in CONFIG_KEEP if config.get(key) is not None}
    meta.update({"agent": agent, "variant": variant, "migrated_from": run_name})
    return meta


def normalize_variant_latest(base: Path, song_id: str, variant: str, *, apply: bool) -> None:
    """把变体级 ``latest.json`` 整成与 workflow 同一个形状。"""
    path = base / "latest.json"
    if not path.is_file():
        return
    payload = read_json(path)
    payload = payload if isinstance(payload, dict) else {}
    rid = payload.get("run")
    if not isinstance(rid, str) or not (base / rid).is_dir():
        return
    if "updated_at" in payload and "variant" not in payload:
        return  # 已经是新形状
    model = payload.get("model")
    if not model:
        for meta_file in sorted((base / rid).glob("sections/*/*/meta.json")):
            meta = read_json(meta_file) or {}
            if meta.get("model"):
                model = meta["model"]
                break
    say(apply, f"{song_id}/{variant}: 规整 latest.json")
    if apply:
        write_json(
            path,
            {
                "run": rid,
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "song": song_id,
                "status": payload.get("status", "completed"),
                "model": model,
                "baseline": variant,
            },
        )


def migrate_paper(song_dir: Path, variant: str, *, apply: bool) -> None:
    base = song_dir / "baselines" / variant
    if not base.is_dir():
        return
    for run in sorted(base.glob("run-*")):
        config = read_json(run / "config.json")
        config = config if isinstance(config, dict) else {}
        sections = run / "sections"
        for sec in sorted(sections.iterdir()) if sections.is_dir() else []:
            if not sec.is_dir():
                continue
            gen, clean = sec / "generation", sec / "cleanup"
            if (gen / "output.json").is_file() and not (sec / "output.json").is_file():
                continue  # 已经迁移过：逐段的旧文件都清了
            section_out = read_json(sec / "output.json") or {}
            align = read_json(sec / "alignment.json") or {}
            section_meta = read_json(sec / "section.json") or {}
            section_id = section_out.get("section") or section_meta.get("id")
            phrases = section_out.get("phrases") or []

            # generation：结论 + 原始正文；旧 meta 的 status/usage 原样保留
            gen_meta = dict(read_json(gen / "meta.json") or {})
            gen_meta.update(_fold_config(config, run.name, variant, "generation"))
            raw = read_text(gen / "response.txt")
            if raw is not None:
                gen_meta["raw_response"] = raw
            write_json(gen / "output.json", {"section": section_id, "source": align.get("source"), "phrases": phrases})
            write_json(gen / "meta.json", gen_meta)

            # cleanup：清理结果 + 原始正文
            clean_meta = dict(read_json(clean / "meta.json") or {})
            clean_meta.update(_fold_config(config, run.name, variant, "cleanup"))
            crawl = read_text(clean / "response.txt")
            if crawl is not None:
                clean_meta["raw_response"] = crawl
            lines: list[str] = []
            if crawl:
                try:
                    payload = json.loads(crawl)
                    if isinstance(payload, dict) and isinstance(payload.get("lines"), list):
                        lines = [x for x in payload["lines"] if isinstance(x, str)]
                except ValueError:
                    lines = []
            write_json(clean / "output.json", {"section": section_id, "lines": lines})
            write_json(clean / "meta.json", clean_meta)

            if apply:
                for path in (
                    sec / "output.json",
                    sec / "section.json",
                    sec / "alignment.json",
                    gen / "response.txt",
                    gen / "response.json",
                    clean / "response.txt",
                    clean / "response.json",
                ):
                    path.unlink(missing_ok=True)
                shutil.rmtree(clean / "attempts", ignore_errors=True)

        say(apply, f"{song_dir.name}/{variant}/{run.name}: 瘦身")
        if not apply:
            continue
        for name in ("config.json", "meta.json", "output.json", "report.json", "song.json"):
            (run / name).unlink(missing_ok=True)
    for name in ("lyrics.md", "lyrics.zh.txt"):
        path = base / name
        if path.exists():
            say(apply, f"{song_dir.name}/{variant}: 删除变体级 {name}")
            if apply:
                path.unlink()
    normalize_variant_latest(base, song_dir.name, variant, apply=apply)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="把旧布局迁移成 <song>/{song.yaml, baselines/, workflow/}")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "data" / "songs")
    parser.add_argument("--song", action="append", default=None, help="只迁移指定歌（可重复）")
    parser.add_argument("--apply", action="store_true", help="真的落盘；默认只预览")
    args = parser.parse_args(argv)
    apply = args.apply

    song_ids = list(dict.fromkeys(args.song)) if args.song else list_songs(root=args.root)
    print(f"共 {len(song_ids)} 首；{'落盘' if apply else '预览'}\n")
    for song_id in song_ids:
        song_dir = args.root / song_id
        song = load_song(song_id, root=args.root)
        changed = migrate_workflow(song, song_dir, args.root, apply=apply)
        if apply and changed:
            cleanup_workflow_leftovers(song_dir, apply=True)
        elif not apply:
            cleanup_workflow_leftovers(song_dir, apply=False)
        for variant in PAPER_VARIANTS:
            migrate_paper(song_dir, variant, apply=apply)

    prompt = REPO_ROOT / "baseline" / "prompt.txt"
    if prompt.exists():
        say(apply, "删除旧 baseline/prompt.txt（旧 prompt、旧粒度）")
        if apply:
            prompt.unlink()

    print("\n完成。" + ("" if apply else "（这是预览，加 --apply 才真的动文件）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
