"""测试用的小工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from dubbing.models import Analysis, Draft, DraftPhrase, Song


def phrase(
    pid: str,
    section: str,
    ja: str,
    n: int | None = None,
    *,
    keep_source: bool = False,
    notes: str | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {"id": pid, "section": section, "ja": ja}
    if keep_source:
        d["keep_source"] = True
    else:
        d["budget"] = {"min": n, "max": n}
    if notes:
        d["notes"] = notes
    return d


def song_json(
    song_id: str,
    sections: list[str],
    phrases: list[dict[str, Any]],
    *,
    title: str = "测试歌",
) -> dict[str, Any]:
    return {
        "song": {"id": song_id, "title": title, "artist": "", "source_lang": "ja"},
        "sections": [{"id": s, "role": "verse"} for s in sections],
        "phrases": phrases,
    }


def make_song(
    song_id: str,
    sections: list[str],
    phrases: list[dict[str, Any]],
    *,
    title: str = "测试歌",
) -> Song:
    return Song.from_json(song_json(song_id, sections, phrases, title=title))


def write_song(
    root: Path,
    song_id: str,
    sections: list[str],
    phrases: list[dict[str, Any]],
    *,
    title: str = "测试歌",
) -> Path:
    d = root / song_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / "song.yaml"
    path.write_text(
        yaml.safe_dump(song_json(song_id, sections, phrases, title=title), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def analysis_json(sections: list[str], **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "overall": {
            "theme": "在夜里找一个人",
            "emotional_arc": "压低 → 挣开",
            "narrative": "独白的散文。",
            "tone_words": ["冷", "钝痛"],
        },
        "imagery_pool": [{"image": "末班车的尾灯", "why": "迟到的告别"}],
        "sections": {s: f"{s} 的情绪" for s in sections},
    }
    data.update(overrides)
    return data


def make_analysis(sections: list[str]) -> Analysis:
    return Analysis.from_json(analysis_json(sections))


def draft_json(section: str, texts: dict[str, str], *, chars: dict[str, int] | None = None) -> dict[str, Any]:
    phrases = []
    for pid, zh in texts.items():
        item: dict[str, Any] = {"id": pid, "zh": zh}
        if chars and pid in chars:
            item["chars"] = chars[pid]
        phrases.append(item)
    return {"section": section, "phrases": phrases}


def make_draft(section: str, texts: dict[str, str], *, chars: dict[str, int] | None = None) -> Draft:
    return Draft.from_json(draft_json(section, texts, chars=chars))


def scripted(*payloads: dict[str, Any]) -> list[str]:
    return [json.dumps(p, ensure_ascii=False) for p in payloads]
